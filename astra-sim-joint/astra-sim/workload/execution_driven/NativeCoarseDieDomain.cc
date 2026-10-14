#include "NativeCoarseDieDomain.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace AstraSim::ExecutionDriven {
namespace {

constexpr std::uint64_t kMaxExactJsonNs = 9007199254740991ULL;
constexpr const char* kDomainSnapshotSchema =
    "native_coarse_die_domain_snapshot_v1";

void require_leg_id(const std::string& leg_id) {
  if (leg_id.empty()) {
    throw std::invalid_argument("invalid_leg_id: leg_id must not be empty");
  }
}

nlohmann::json owner_json(const coarse_die::Owner& owner) {
  return {{"request_id", owner.request_id},
          {"node_id", owner.node_id},
          {"execution_id", owner.execution_id},
          {"generation", owner.generation},
          {"key", owner.key()}};
}

bool same_owner(const coarse_die::Owner& left,
                const coarse_die::Owner& right) {
  return left.request_id == right.request_id &&
         left.node_id == right.node_id &&
         left.execution_id == right.execution_id &&
         left.generation == right.generation;
}

std::uint64_t parse_u64(const nlohmann::json& value, const char* field) {
  if (!value.is_number_integer()) {
    throw std::invalid_argument(std::string("invalid_snapshot: ") + field +
                                " must be an integer");
  }
  if (value.is_number_unsigned()) return value.get<std::uint64_t>();
  const std::int64_t signed_value = value.get<std::int64_t>();
  if (signed_value < 0) {
    throw std::invalid_argument(std::string("invalid_snapshot: ") + field +
                                " must be nonnegative");
  }
  return static_cast<std::uint64_t>(signed_value);
}

std::uint64_t parse_timestamp(const nlohmann::json& value,
                             const char* field) {
  const std::uint64_t timestamp = parse_u64(value, field);
  if (timestamp > kMaxExactJsonNs) {
    throw std::invalid_argument(std::string("invalid_snapshot: ") + field +
                                " exceeds the exact JSON nanosecond range");
  }
  return timestamp;
}

coarse_die::Owner parse_owner(const nlohmann::json& value) {
  if (!value.is_object() || value.size() != 5 ||
      !value.contains("request_id") || !value.contains("node_id") ||
      !value.contains("execution_id") || !value.contains("generation") ||
      !value.contains("key")) {
    throw std::invalid_argument(
        "invalid_snapshot: owner must contain its four ids and derived key");
  }
  coarse_die::Owner owner;
  owner.request_id = parse_u64(value.at("request_id"), "owner.request_id");
  owner.node_id = parse_u64(value.at("node_id"), "owner.node_id");
  owner.execution_id =
      parse_u64(value.at("execution_id"), "owner.execution_id");
  owner.generation = parse_u64(value.at("generation"), "owner.generation");
  if (owner.generation == 0 || !value.at("key").is_string() ||
      value.at("key").get<std::string>() != owner.key()) {
    throw std::invalid_argument(
        "invalid_snapshot: owner generation or derived key is invalid");
  }
  return owner;
}

void require_finite_json(const nlohmann::json& value, const char* field) {
  if (value.is_number_float() && !std::isfinite(value.get<double>())) {
    throw std::invalid_argument(std::string("invalid_snapshot: nonfinite ") +
                                field);
  }
  if (value.is_array()) {
    for (const auto& child : value) require_finite_json(child, field);
  } else if (value.is_object()) {
    for (const auto& [key, child] : value.items()) {
      (void)key;
      require_finite_json(child, field);
    }
  }
}

void require_keys(const nlohmann::json& value,
                  std::initializer_list<const char*> keys,
                  const char* field) {
  if (!value.is_object() || value.size() != keys.size()) {
    throw std::invalid_argument(std::string("invalid_snapshot: ") + field +
                                " has an invalid object shape");
  }
  for (const char* key : keys) {
    if (!value.contains(key)) {
      throw std::invalid_argument(std::string("invalid_snapshot: ") + field +
                                  " is missing " + key);
    }
  }
}

bool same_owner_json(const nlohmann::json& value,
                     const coarse_die::Owner& expected) {
  try {
    const auto actual = parse_owner(value);
    return same_owner(actual, expected);
  } catch (const std::exception&) {
    return false;
  }
}

void validate_progress(const nlohmann::json& progress,
                       const coarse_die::Owner& owner,
                       const std::string& plan_fingerprint,
                       const std::string& hardware_fingerprint,
                       std::uint64_t expected_handle,
                       std::uint64_t expected_launch_ns,
                       bool terminal, bool cancelled,
                       std::optional<std::uint64_t> finish_ns,
                       std::uint64_t now_ns,
                       std::set<std::uint64_t>& observed_handles) {
  require_finite_json(progress, "execution progress");
  if (!progress.is_object() || !progress.contains("handle") ||
      !progress.contains("owner") || !progress.contains("state") ||
      !progress.contains("issue_ns") || !progress.contains("ready_ns") ||
      !progress.contains("finish_ns") || !progress.contains("terminal") ||
      !progress.contains("cancelled") ||
      !progress.contains("plan_fingerprint") ||
      !progress.contains("hardware_fingerprint")) {
    throw std::invalid_argument(
        "invalid_snapshot: retained progress is incomplete");
  }
  const std::uint64_t handle = parse_u64(progress.at("handle"),
                                         "progress.handle");
  const std::uint64_t issue_ns =
      parse_timestamp(progress.at("issue_ns"), "progress.issue_ns");
  const std::uint64_t ready_ns =
      parse_timestamp(progress.at("ready_ns"), "progress.ready_ns");
  if (handle == 0 || (expected_handle != 0 && handle != expected_handle) ||
      !observed_handles.insert(handle).second || issue_ns > now_ns ||
      ready_ns < issue_ns || ready_ns - issue_ns != expected_launch_ns ||
      !same_owner_json(progress.at("owner"), owner) ||
      !progress.at("terminal").is_boolean() ||
      progress.at("terminal").get<bool>() != terminal ||
      !progress.at("cancelled").is_boolean() ||
      progress.at("cancelled").get<bool>() != cancelled ||
      !progress.at("plan_fingerprint").is_string() ||
      progress.at("plan_fingerprint").get<std::string>() != plan_fingerprint ||
      !progress.at("hardware_fingerprint").is_string() ||
      progress.at("hardware_fingerprint").get<std::string>() !=
          hardware_fingerprint) {
    throw std::invalid_argument(
        "invalid_snapshot: retained progress identity differs from its leg");
  }
  const auto& saved_finish = progress.at("finish_ns");
  if (terminal) {
    if (!finish_ns || saved_finish.is_null() ||
        parse_timestamp(saved_finish, "progress.finish_ns") != *finish_ns ||
        *finish_ns < issue_ns || *finish_ns > now_ns ||
        !progress.at("state").is_string() ||
        progress.at("state").get<std::string>() !=
            (cancelled ? "cancelled" : "finished")) {
      throw std::invalid_argument(
          "invalid_snapshot: terminal progress time/state is inconsistent");
    }
  } else {
    if (finish_ns || !saved_finish.is_null() || cancelled ||
        !progress.at("state").is_string()) {
      throw std::invalid_argument(
          "invalid_snapshot: active progress lifecycle is inconsistent");
    }
    const std::string state = progress.at("state").get<std::string>();
    if (state != "queued" && state != "running") {
      throw std::invalid_argument(
          "invalid_snapshot: active progress has a non-live state");
    }
  }
}

}  // namespace

NativeCoarseDieDomain::NativeCoarseDieDomain(
    std::string resource_key, coarse_die::Hardware hardware)
    : NativeCoarseDieDomain(
          std::move(resource_key),
          std::make_shared<const coarse_die::Hardware>(std::move(hardware))) {}

NativeCoarseDieDomain::NativeCoarseDieDomain(
    std::string resource_key,
    std::shared_ptr<const coarse_die::Hardware> hardware)
    : resource_key_(std::move(resource_key)),
      hardware_(std::move(hardware)),
      runtime_(hardware_) {
  if (resource_key_.empty()) {
    throw std::invalid_argument(
        "invalid_resource_key: coarse die domain key must not be empty");
  }
}

NativeCoarseDieDomain::NativeCoarseDieDomain(
    std::string resource_key,
    std::shared_ptr<const coarse_die::Hardware> hardware,
    const coarse_die::Runtime& runtime)
    : resource_key_(std::move(resource_key)),
      hardware_(std::move(hardware)),
      runtime_(runtime) {}

void NativeCoarseDieDomain::validate_arrival(std::uint64_t now_ns) const {
  if (now_ns > kMaxExactJsonNs) {
    throw std::invalid_argument(
        "invalid_time: coarse die timestamps exceed the exact JSON integer range");
  }
  if (now_ns < runtime_.now_ns()) {
    throw std::invalid_argument(
        "invalid_time: coarse die domain arrival moved backwards");
  }
}

void NativeCoarseDieDomain::begin_plan_chain(
    std::string leg_id,
    std::vector<std::shared_ptr<const coarse_die::Plan>> plans,
    coarse_die::Owner owner, std::uint64_t now_ns) {
  require_leg_id(leg_id);
  if (plans.empty()) {
    throw std::invalid_argument(
        "invalid_plan_chain: at least one Plan is required");
  }
  validate_arrival(now_ns);
  for (const auto& plan : plans) {
    if (!plan) {
      throw std::invalid_argument("invalid_plan_chain: null Plan");
    }
    if (plan->hardware_fingerprint != hardware_->fingerprint) {
      throw std::invalid_argument(
          "incompatible_fingerprint: Plan hardware fingerprint mismatch");
    }
    if (plan->launch_ns > kMaxExactJsonNs - now_ns) {
      throw std::invalid_argument(
          "invalid_time: Plan launch exceeds exact JSON nanosecond range");
    }
    // Validate every Plan before advancing this domain. The canonical wire
    // serializer checks the public Plan structure and fingerprint; a private
    // Runtime additionally checks hardware-specific resource use without
    // mutating this domain or estimating latency.
    (void)coarse_die::plan_to_json(*plan);
    coarse_die::Runtime validator(hardware_);
    (void)validator.submit(plan, owner, now_ns);
  }
  if (entries_.find(leg_id) != entries_.end()) {
    throw std::invalid_argument(
        "duplicate_leg_id: retire the previous leg before reusing its id");
  }
  // Existing work is settled through every public Runtime event first. This
  // lets any earlier chain submit its successor at its actual terminal tick
  // before this arrival is admitted.
  advance_to(now_ns);

  Entry entry;
  entry.kind = EntryKind::PlanChain;
  entry.owner = owner;
  entry.plans = std::move(plans);
  auto inserted = entries_.emplace(std::move(leg_id), std::move(entry));
  if (!inserted.second) {
    throw std::invalid_argument("duplicate_leg_id: leg_id is already present");
  }
  try {
    submit_current(inserted.first->first, inserted.first->second, now_ns);
  } catch (...) {
    entries_.erase(inserted.first);
    throw;
  }
  settle_terminals();
}

void NativeCoarseDieDomain::begin_endpoint(
    std::string leg_id, const nlohmann::json& endpoint,
    coarse_die::Owner owner, std::uint64_t now_ns) {
  require_leg_id(leg_id);
  if (entries_.find(leg_id) != entries_.end()) {
    throw std::invalid_argument(
        "duplicate_leg_id: retire the previous leg before reusing its id");
  }
  validate_arrival(now_ns);

  // Parse, validate hardware use, and test endpoint identity against a
  // disposable Runtime before this domain advances. Track the small public
  // endpoint/transaction binding state locally so duplicate and owner
  // conflicts are rejected without copying the whole live Runtime.
  coarse_die::Runtime validator(hardware_);
  (void)validator.submit_endpoint(endpoint, owner, now_ns);
  const std::uint64_t endpoint_id =
      endpoint.at("endpoint_id").get<std::uint64_t>();
  const std::uint64_t transaction_id =
      endpoint.at("transaction_id").get<std::uint64_t>();
  if (active_endpoint_ids_.find(endpoint_id) != active_endpoint_ids_.end()) {
    throw std::invalid_argument(
        "duplicate_endpoint_id: endpoint_id is already live in this domain");
  }
  const auto transaction =
      active_endpoint_transactions_.find(transaction_id);
  if (transaction != active_endpoint_transactions_.end() &&
      !same_owner(transaction->second.owner, owner)) {
    throw std::invalid_argument(
        "transaction_owner_conflict: transaction_id belongs to another owner");
  }
  if (transaction != active_endpoint_transactions_.end() &&
      transaction->second.live_endpoints ==
          std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error(
        "overflow: endpoint transaction reference count exhausted");
  }
  const std::uint64_t initial_delay_ns = endpoint.contains("initial_delay_ns")
      ? endpoint.at("initial_delay_ns").get<std::uint64_t>()
      : 0;
  if (initial_delay_ns > kMaxExactJsonNs - now_ns) {
    throw std::invalid_argument(
        "invalid_time: endpoint launch exceeds exact JSON nanosecond range");
  }
  advance_to(now_ns);

  Entry entry;
  entry.kind = EntryKind::Endpoint;
  entry.owner = owner;
  entry.endpoint = endpoint;
  entry.endpoint_id = endpoint_id;
  entry.transaction_id = transaction_id;
  auto inserted = entries_.emplace(std::move(leg_id), std::move(entry));
  if (!inserted.second) {
    throw std::invalid_argument("duplicate_leg_id: leg_id is already present");
  }
  bool endpoint_id_bound = false;
  bool transaction_bound = false;
  try {
    submit_current(inserted.first->first, inserted.first->second, now_ns);
    const auto id_binding =
        active_endpoint_ids_.emplace(endpoint_id, inserted.first->first);
    if (!id_binding.second) {
      throw std::invalid_argument(
          "duplicate_endpoint_id: endpoint_id is already live in this domain");
    }
    endpoint_id_bound = true;
    auto tx = active_endpoint_transactions_.find(transaction_id);
    if (tx == active_endpoint_transactions_.end()) {
      const auto tx_binding = active_endpoint_transactions_.emplace(
          transaction_id, EndpointTransactionOwner{owner, 1});
      if (!tx_binding.second) {
        throw std::logic_error(
            "invalid_wrapper_state: endpoint transaction binding raced");
      }
    } else {
      ++tx->second.live_endpoints;
    }
    transaction_bound = true;
  } catch (...) {
    if (transaction_bound) {
      auto tx = active_endpoint_transactions_.find(transaction_id);
      if (tx != active_endpoint_transactions_.end() &&
          same_owner(tx->second.owner, owner)) {
        if (tx->second.live_endpoints > 1) --tx->second.live_endpoints;
        else active_endpoint_transactions_.erase(tx);
      }
    }
    if (endpoint_id_bound) active_endpoint_ids_.erase(endpoint_id);
    if (inserted.first->second.handle.has_value()) {
      const auto handle = *inserted.first->second.handle;
      if (!runtime_.terminal_finish_ns(handle).has_value()) {
        runtime_.cancel(handle, now_ns);
      }
      runtime_.destroy_execution(handle);
    }
    entries_.erase(inserted.first);
    throw;
  }
  settle_terminals();
}

void NativeCoarseDieDomain::submit_current(const std::string& leg_id,
                                          Entry& entry,
                                          std::uint64_t now_ns) {
  (void)leg_id;
  if (entry.handle.has_value() || entry.terminal) {
    throw std::logic_error(
        "invalid_wrapper_state: cannot submit an active or terminal entry");
  }
  if (entry.kind == EntryKind::Endpoint) {
    entry.handle = runtime_.submit_endpoint(entry.endpoint, entry.owner, now_ns);
    return;
  }
  if (entry.plan_index >= entry.plans.size()) {
    throw std::logic_error("invalid_wrapper_state: Plan chain cursor is out of range");
  }
  entry.handle = runtime_.submit(entry.plans[entry.plan_index], entry.owner,
                                 now_ns);
}

void NativeCoarseDieDomain::record_terminal(const std::string& leg_id,
                                            Entry& entry,
                                            std::uint64_t finish_ns,
                                            bool cancelled) {
  if (entry.terminal) return;
  completions_.push_back({leg_id, finish_ns, cancelled});
  entry.terminal = true;
  entry.cancelled = cancelled;
  entry.finish_ns = finish_ns;
}

void NativeCoarseDieDomain::settle_terminals() {
  bool changed = true;
  while (changed) {
    changed = false;
    // entries_ is ordered by leg_id, so chains terminal on the same Runtime
    // tick publish successors in a stable order independent of arrival order.
    for (auto& [leg_id, entry] : entries_) {
      if (entry.terminal || !entry.handle.has_value()) continue;
      const std::uint64_t handle = *entry.handle;
      const auto finish = runtime_.terminal_finish_ns(handle);
      if (!finish.has_value()) continue;

      // Preserve exact native progress while removing the live Runtime handle
      // (including its endpoint registry entry) immediately at terminal.
      nlohmann::json terminal_progress = runtime_.query_progress(handle);
      const bool was_cancelled =
          terminal_progress.value("cancelled", false);
      runtime_.destroy_execution(handle);
      if (entry.kind == EntryKind::Endpoint) {
        release_endpoint_binding(entry);
      }
      entry.handle.reset();
      entry.last_progress = terminal_progress;

      if (entry.kind == EntryKind::Endpoint) {
        record_terminal(leg_id, entry, *finish, was_cancelled);
      } else {
        if (!was_cancelled) {
          entry.completed_plan_progress.push_back(terminal_progress);
        }
        if (!was_cancelled && entry.plan_index + 1 < entry.plans.size()) {
          ++entry.plan_index;
          submit_current(leg_id, entry, *finish);
        } else {
          record_terminal(leg_id, entry, *finish, was_cancelled);
        }
      }
      changed = true;
    }
  }
}

void NativeCoarseDieDomain::release_endpoint_binding(const Entry& entry) {
  active_endpoint_ids_.erase(entry.endpoint_id);
  auto tx = active_endpoint_transactions_.find(entry.transaction_id);
  if (tx == active_endpoint_transactions_.end()) {
    throw std::logic_error(
        "invalid_wrapper_state: endpoint transaction binding is missing");
  }
  if (tx->second.live_endpoints > 1) {
    --tx->second.live_endpoints;
  } else {
    active_endpoint_transactions_.erase(tx);
  }
}

void NativeCoarseDieDomain::advance_to(std::uint64_t absolute_ns) {
  validate_arrival(absolute_ns);
  while (true) {
    const auto next = runtime_.next_event();
    if (!next.has_value() || *next > absolute_ns) break;
    if (*next <= runtime_.now_ns()) {
      throw std::runtime_error(
          "numerical_failure: coarse die next_event did not advance time");
    }
    runtime_.advance_to(*next);
    settle_terminals();
  }
  if (runtime_.now_ns() < absolute_ns) {
    runtime_.advance_to(absolute_ns);
    settle_terminals();
  } else {
    settle_terminals();
  }
}

void NativeCoarseDieDomain::update_capacity(
    std::size_t resource, double absolute_physical_rate,
    std::uint64_t now_ns) {
  // Reject every caller-controlled error before advancing this live Domain.
  // Keep resource/rate/disabled-resource validation canonical by asking an
  // empty Runtime with the same Hardware to validate the proposed capacity.
  validate_arrival(now_ns);
  coarse_die::Runtime validator(hardware_);
  validator.update_capacity(resource, absolute_physical_rate, 0);

  // Runtime events at this tick (including terminal Plans and chain
  // successors) are settled using the old capacity. Then the actual Runtime
  // integrates through now_ns and installs the new physical rate.
  advance_to(now_ns);
  runtime_.update_capacity(resource, absolute_physical_rate, now_ns);
  settle_terminals();
}

std::optional<std::uint64_t> NativeCoarseDieDomain::next_event() const {
  const auto next = runtime_.next_event();
  if (next.has_value() && *next > kMaxExactJsonNs) {
    throw std::overflow_error(
        "unsupported_time: next coarse die event exceeds exact JSON integer range");
  }
  return next;
}

std::vector<NativeCoarseDieCompletion>
NativeCoarseDieDomain::take_completions() {
  std::vector<NativeCoarseDieCompletion> out;
  out.swap(completions_);
  return out;
}

bool NativeCoarseDieDomain::cancel(const std::string& leg_id,
                                   std::uint64_t now_ns) {
  const auto found = entries_.find(leg_id);
  if (found == entries_.end()) return false;
  validate_arrival(now_ns);
  advance_to(now_ns);

  auto current = entries_.find(leg_id);
  if (current == entries_.end() || current->second.terminal ||
      !current->second.handle.has_value()) {
    return false;
  }
  runtime_.cancel(*current->second.handle, now_ns);
  settle_terminals();
  return current->second.terminal && current->second.cancelled;
}

nlohmann::json NativeCoarseDieDomain::progress(
    const std::string& leg_id) const {
  const auto found = entries_.find(leg_id);
  if (found == entries_.end()) {
    throw std::out_of_range("unknown_leg_id: no coarse die leg is retained");
  }
  const Entry& entry = found->second;
  nlohmann::json active_progress = nullptr;
  if (entry.handle.has_value()) {
    active_progress = runtime_.query_progress(*entry.handle);
  }
  const char* kind = entry.kind == EntryKind::PlanChain ? "plan_chain"
                                                        : "endpoint";
  const char* state = entry.cancelled ? "cancelled"
                      : entry.terminal ? "finished"
                      : entry.handle.has_value() ? "running"
                                                 : "queued";
  nlohmann::json result = {
      {"resource_key", resource_key_},
      {"leg_id", leg_id},
      {"kind", kind},
      {"state", state},
      {"terminal", entry.terminal},
      {"cancelled", entry.cancelled},
      {"finish_ns", entry.terminal ? nlohmann::json(entry.finish_ns)
                                   : nlohmann::json(nullptr)},
      {"owner", owner_json(entry.owner)},
      {"plan_index", entry.plan_index},
      {"plan_count", entry.plans.size()},
      {"completed_plan_progress", entry.completed_plan_progress},
      {"active_execution", active_progress},
      {"last_progress", entry.last_progress.is_null()
                             ? nlohmann::json(nullptr)
                             : entry.last_progress},
      {"runtime_now_ns", runtime_.now_ns()}};
  if (entry.kind == EntryKind::Endpoint) result["endpoint"] = entry.endpoint;
  return result;
}

bool NativeCoarseDieDomain::retire(const std::string& leg_id) {
  const auto found = entries_.find(leg_id);
  if (found == entries_.end() || !found->second.terminal) return false;
  for (const auto& completion : completions_) {
    if (completion.leg_id == leg_id) return false;
  }
  entries_.erase(found);
  return true;
}

nlohmann::json NativeCoarseDieDomain::counters() const {
  return runtime_.counters();
}

nlohmann::json NativeCoarseDieDomain::export_snapshot() const {
  nlohmann::json entries = nlohmann::json::array();
  for (const auto& [leg_id, entry] : entries_) {
    const char* kind = entry.kind == EntryKind::PlanChain ? "plan_chain"
                                                          : "endpoint";
    nlohmann::json wire = {
        {"leg_id", leg_id},
        {"kind", kind},
        {"owner", owner_json(entry.owner)},
        {"handle", entry.handle ? nlohmann::json(*entry.handle)
                                 : nlohmann::json(nullptr)},
        {"terminal", entry.terminal},
        {"cancelled", entry.cancelled},
        {"finish_ns", entry.terminal ? nlohmann::json(entry.finish_ns)
                                     : nlohmann::json(nullptr)},
        {"last_progress", entry.last_progress},
        {"plan_index", entry.plan_index},
        {"completed_plan_progress", entry.completed_plan_progress}};
    if (entry.kind == EntryKind::PlanChain) {
      nlohmann::json plans = nlohmann::json::array();
      for (const auto& plan : entry.plans) {
        plans.push_back(coarse_die::plan_to_json(*plan));
      }
      wire["plans"] = std::move(plans);
    } else {
      wire["endpoint"] = entry.endpoint;
      wire["endpoint_id"] = entry.endpoint_id;
      wire["transaction_id"] = entry.transaction_id;
    }
    entries.push_back(std::move(wire));
  }

  nlohmann::json endpoint_ids = nlohmann::json::array();
  for (const auto& [endpoint_id, leg_id] : active_endpoint_ids_) {
    endpoint_ids.push_back({{"endpoint_id", endpoint_id}, {"leg_id", leg_id}});
  }
  nlohmann::json transactions = nlohmann::json::array();
  for (const auto& [transaction_id, record] : active_endpoint_transactions_) {
    transactions.push_back({{"transaction_id", transaction_id},
                            {"owner", owner_json(record.owner)},
                            {"live_endpoints", record.live_endpoints}});
  }
  nlohmann::json completions = nlohmann::json::array();
  for (const auto& completion : completions_) {
    completions.push_back({{"leg_id", completion.leg_id},
                           {"finish_ns", completion.finish_ns},
                           {"cancelled", completion.cancelled}});
  }
  return {{"schema", kDomainSnapshotSchema},
          {"snapshot_version", 1},
          {"resource_key", resource_key_},
          {"hardware_fingerprint", hardware_->fingerprint},
          {"runtime", runtime_.export_snapshot()},
          {"entries", std::move(entries)},
          {"endpoint_bindings",
           {{"endpoint_ids", std::move(endpoint_ids)},
            {"transactions", std::move(transactions)}}},
          {"completion_queue", std::move(completions)}};
}

std::unique_ptr<NativeCoarseDieDomain>
NativeCoarseDieDomain::import_snapshot(
    std::string resource_key,
    std::shared_ptr<const coarse_die::Hardware> hardware,
    const nlohmann::json& snapshot) {
  require_keys(snapshot,
               {"schema", "snapshot_version", "resource_key",
                "hardware_fingerprint", "runtime", "entries",
                "endpoint_bindings", "completion_queue"},
               "domain snapshot");
  require_finite_json(snapshot, "domain snapshot");
  if (snapshot.at("schema") != kDomainSnapshotSchema ||
      parse_u64(snapshot.at("snapshot_version"), "snapshot_version") != 1) {
    throw std::invalid_argument(
        "incompatible_schema: expected native_coarse_die_domain_snapshot_v1");
  }
  if (resource_key.empty() || !hardware) {
    throw std::invalid_argument(
        "invalid_snapshot: resource key and Hardware are required");
  }
  if (!snapshot.at("resource_key").is_string() ||
      snapshot.at("resource_key").get<std::string>() != resource_key) {
    throw std::invalid_argument(
        "incompatible_resource: domain snapshot resource key differs");
  }
  if (!snapshot.at("hardware_fingerprint").is_string() ||
      snapshot.at("hardware_fingerprint").get<std::string>() !=
          hardware->fingerprint) {
    throw std::invalid_argument(
        "incompatible_fingerprint: domain hardware fingerprint differs");
  }

  const auto& runtime_wire = snapshot.at("runtime");
  if (!runtime_wire.is_object() ||
      runtime_wire.value("schema", std::string()) !=
          "coarse_die_snapshot_v3" ||
      !runtime_wire.contains("snapshot_version") ||
      parse_u64(runtime_wire.at("snapshot_version"),
                "runtime.snapshot_version") != 3 ||
      !runtime_wire.contains("hardware_fingerprint") ||
      runtime_wire.at("hardware_fingerprint") != hardware->fingerprint) {
    throw std::invalid_argument(
        "incompatible_schema: domain requires a matching coarse_die_snapshot_v3 Runtime");
  }
  const std::uint64_t now_ns =
      parse_timestamp(runtime_wire.at("now_ns"), "runtime.now_ns");
  const std::uint64_t next_handle =
      parse_u64(runtime_wire.at("next_handle"), "runtime.next_handle");

  if (!runtime_wire.at("executions").is_array() ||
      !snapshot.at("entries").is_array() ||
      !snapshot.at("completion_queue").is_array()) {
    throw std::invalid_argument(
        "invalid_snapshot: Runtime executions, entries, and completion queue must be arrays");
  }
  std::map<std::uint64_t, const nlohmann::json*> runtime_executions;
  for (const auto& execution : runtime_wire.at("executions")) {
    if (!execution.is_object() || !execution.contains("handle")) {
      throw std::invalid_argument(
          "invalid_snapshot: Runtime execution has no handle");
    }
    const std::uint64_t handle =
        parse_u64(execution.at("handle"), "runtime.execution.handle");
    const std::uint64_t issue_ns =
        parse_timestamp(execution.at("issue_ns"),
                        "runtime.execution.issue_ns");
    const std::uint64_t ready_ns =
        parse_timestamp(execution.at("ready_ns"),
                        "runtime.execution.ready_ns");
    if (issue_ns > now_ns || ready_ns < issue_ns) {
      throw std::invalid_argument(
          "invalid_snapshot: Runtime execution issue/ready times are inconsistent");
    }
    if (!execution.at("finish_ns").is_null()) {
      const std::uint64_t finish_ns = parse_timestamp(
          execution.at("finish_ns"), "runtime.execution.finish_ns");
      if (finish_ns < issue_ns || finish_ns > now_ns) {
        throw std::invalid_argument(
            "invalid_snapshot: Runtime execution finish time is inconsistent");
      }
    }
    if (handle == 0 || !runtime_executions.emplace(handle, &execution).second) {
      throw std::invalid_argument(
          "invalid_snapshot: duplicate or zero Runtime handle");
    }
  }

  std::map<std::string, Entry> imported_entries;
  std::map<std::uint64_t, std::string> expected_endpoint_ids;
  std::map<std::uint64_t, EndpointTransactionOwner>
      expected_endpoint_transactions;
  std::set<std::uint64_t> observed_history_handles;
  std::optional<std::string> previous_leg_id;

  for (const auto& wire : snapshot.at("entries")) {
    if (!wire.is_object() || !wire.contains("kind") ||
        !wire.at("kind").is_string()) {
      throw std::invalid_argument(
          "invalid_snapshot: domain entry has no supported kind");
    }
    const std::string kind = wire.at("kind").get<std::string>();
    if (kind == "plan_chain") {
      require_keys(wire,
                   {"leg_id", "kind", "owner", "handle", "terminal",
                    "cancelled", "finish_ns", "last_progress", "plan_index",
                    "completed_plan_progress", "plans"},
                   "plan-chain entry");
    } else if (kind == "endpoint") {
      require_keys(wire,
                   {"leg_id", "kind", "owner", "handle", "terminal",
                    "cancelled", "finish_ns", "last_progress", "plan_index",
                    "completed_plan_progress", "endpoint", "endpoint_id",
                    "transaction_id"},
                   "endpoint entry");
    } else {
      throw std::invalid_argument(
          "invalid_snapshot: unknown domain entry kind");
    }

    if (!wire.at("leg_id").is_string()) {
      throw std::invalid_argument(
          "invalid_snapshot: leg_id must be a string");
    }
    const std::string leg_id = wire.at("leg_id").get<std::string>();
    require_leg_id(leg_id);
    if (previous_leg_id && leg_id <= *previous_leg_id) {
      throw std::invalid_argument(
          "invalid_snapshot: entries must have unique sorted leg ids");
    }
    previous_leg_id = leg_id;

    Entry entry;
    entry.kind = kind == "plan_chain" ? EntryKind::PlanChain
                                        : EntryKind::Endpoint;
    entry.owner = parse_owner(wire.at("owner"));
    if (!wire.at("terminal").is_boolean() ||
        !wire.at("cancelled").is_boolean()) {
      throw std::invalid_argument(
          "invalid_snapshot: entry terminal flags must be booleans");
    }
    entry.terminal = wire.at("terminal").get<bool>();
    entry.cancelled = wire.at("cancelled").get<bool>();
    if (entry.cancelled && !entry.terminal) {
      throw std::invalid_argument(
          "invalid_snapshot: a nonterminal entry cannot be cancelled");
    }
    if (wire.at("handle").is_null()) {
      entry.handle.reset();
    } else {
      entry.handle = parse_u64(wire.at("handle"), "entry.handle");
      if (*entry.handle == 0) {
        throw std::invalid_argument(
            "invalid_snapshot: execution handle must be positive");
      }
    }
    if (entry.terminal != !wire.at("finish_ns").is_null() ||
        entry.terminal != !entry.handle.has_value()) {
      throw std::invalid_argument(
          "invalid_snapshot: entry handle and terminal timestamp disagree");
    }
    if (entry.terminal) {
      entry.finish_ns = parse_timestamp(wire.at("finish_ns"),
                                        "entry.finish_ns");
      if (entry.finish_ns > now_ns) {
        throw std::invalid_argument(
            "invalid_snapshot: entry finishes after Runtime now_ns");
      }
    }
    entry.last_progress = wire.at("last_progress");
    if (!wire.at("completed_plan_progress").is_array()) {
      throw std::invalid_argument(
          "invalid_snapshot: completed_plan_progress must be an array");
    }
    for (const auto& progress : wire.at("completed_plan_progress")) {
      entry.completed_plan_progress.push_back(progress);
    }
    const std::uint64_t cursor =
        parse_u64(wire.at("plan_index"), "entry.plan_index");
    if (cursor > std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument(
          "invalid_snapshot: Plan cursor exceeds size_t");
    }
    entry.plan_index = static_cast<std::size_t>(cursor);

    if (entry.kind == EntryKind::PlanChain) {
      if (!wire.at("plans").is_array() || wire.at("plans").empty()) {
        throw std::invalid_argument(
            "invalid_snapshot: Plan chain must retain at least one Plan");
      }
      for (const auto& plan_wire : wire.at("plans")) {
        require_finite_json(plan_wire, "Plan wire");
        const auto plan = coarse_die::plan_from_json(plan_wire);
        if (!plan || plan->hardware_fingerprint != hardware->fingerprint ||
            coarse_die::plan_to_json(*plan) != plan_wire ||
            plan->launch_ns > kMaxExactJsonNs) {
          throw std::invalid_argument(
              "invalid_snapshot: Plan is noncanonical or incompatible with Hardware");
        }
        // Runtime::submit is the public hardware-specific structural and
        // resource validator. It does not run latency formulas or mutate the
        // imported Runtime.
        coarse_die::Runtime validator(hardware);
        (void)validator.submit(plan, entry.owner, 0);
        entry.plans.push_back(plan);
      }
      if (entry.plan_index >= entry.plans.size()) {
        throw std::invalid_argument(
            "invalid_snapshot: Plan cursor is out of range");
      }
      const std::size_t expected_completed =
          entry.plan_index + ((entry.terminal && !entry.cancelled) ? 1 : 0);
      if (entry.completed_plan_progress.size() != expected_completed ||
          (entry.terminal && !entry.cancelled &&
           entry.plan_index + 1 != entry.plans.size()) ||
          (entry.terminal && entry.cancelled &&
           entry.plan_index >= entry.plans.size())) {
        throw std::invalid_argument(
            "invalid_snapshot: Plan cursor and retained completion history disagree");
      }
      for (std::size_t index = 0;
           index < entry.completed_plan_progress.size(); ++index) {
        const auto& progress = entry.completed_plan_progress[index];
        const std::uint64_t progress_finish_ns = parse_timestamp(
            progress.at("finish_ns"), "completed_plan_progress.finish_ns");
        validate_progress(progress, entry.owner,
                          entry.plans[index]->fingerprint,
                          hardware->fingerprint, 0,
                          entry.plans[index]->launch_ns, true, false,
                          progress_finish_ns, now_ns,
                          observed_history_handles);
        const std::uint64_t finish_ns = progress_finish_ns;
        if (index > 0) {
          const std::uint64_t prior_finish = parse_timestamp(
              entry.completed_plan_progress[index - 1].at("finish_ns"),
              "completed_plan_progress.finish_ns");
          const std::uint64_t prior_handle = parse_u64(
              entry.completed_plan_progress[index - 1].at("handle"),
              "completed_plan_progress.handle");
          if (parse_timestamp(progress.at("issue_ns"),
                              "completed_plan_progress.issue_ns") !=
                  prior_finish ||
              parse_u64(progress.at("handle"),
                        "completed_plan_progress.handle") <= prior_handle) {
            throw std::invalid_argument(
                "invalid_snapshot: chain successor tick/handle order is inconsistent");
          }
        }
        if (finish_ns > now_ns) {
          throw std::invalid_argument(
              "invalid_snapshot: completed Plan is after Runtime now_ns");
        }
      }
      if (entry.last_progress.is_null()) {
        if (entry.plan_index != 0 || entry.terminal) {
          throw std::invalid_argument(
              "invalid_snapshot: chain is missing its prior or terminal progress");
        }
      } else if (!entry.completed_plan_progress.empty() &&
                 !entry.cancelled) {
        if (entry.last_progress != entry.completed_plan_progress.back()) {
          throw std::invalid_argument(
              "invalid_snapshot: last_progress differs from completed chain tail");
        }
      } else if (entry.terminal && entry.cancelled) {
        validate_progress(entry.last_progress, entry.owner,
                          entry.plans[entry.plan_index]->fingerprint,
                          hardware->fingerprint, 0,
                          entry.plans[entry.plan_index]->launch_ns, true, true,
                          entry.finish_ns, now_ns, observed_history_handles);
        if (entry.plan_index > 0 &&
            parse_timestamp(entry.last_progress.at("issue_ns"),
                            "last_progress.issue_ns") !=
                parse_timestamp(
                    entry.completed_plan_progress.back().at("finish_ns"),
                    "completed_plan_progress.finish_ns")) {
          throw std::invalid_argument(
              "invalid_snapshot: cancelled Plan did not start at prior terminal tick");
        }
        if (entry.plan_index > 0 &&
            parse_u64(entry.last_progress.at("handle"),
                      "last_progress.handle") <=
                parse_u64(entry.completed_plan_progress.back().at("handle"),
                          "completed_plan_progress.handle")) {
          throw std::invalid_argument(
              "invalid_snapshot: cancelled Plan handle does not follow chain history");
        }
      } else {
        throw std::invalid_argument(
            "invalid_snapshot: chain last_progress has no matching history");
      }
      if (entry.terminal && !entry.cancelled &&
          (entry.last_progress != entry.completed_plan_progress.back() ||
           parse_timestamp(entry.last_progress.at("finish_ns"),
                           "last_progress.finish_ns") != entry.finish_ns)) {
        throw std::invalid_argument(
            "invalid_snapshot: successful terminal differs from final Plan progress");
      }
    } else {
      if (entry.plan_index != 0 ||
          !entry.completed_plan_progress.empty() ||
          !wire.at("endpoint").is_object()) {
        throw std::invalid_argument(
            "invalid_snapshot: endpoint entry contains Plan-chain state");
      }
      entry.endpoint = wire.at("endpoint");
      entry.endpoint_id = parse_u64(wire.at("endpoint_id"),
                                    "entry.endpoint_id");
      entry.transaction_id = parse_u64(wire.at("transaction_id"),
                                       "entry.transaction_id");
      if (parse_u64(entry.endpoint.value("endpoint_id", nlohmann::json()),
                    "endpoint.endpoint_id") != entry.endpoint_id ||
          parse_u64(entry.endpoint.value("transaction_id", nlohmann::json()),
                    "endpoint.transaction_id") != entry.transaction_id) {
        throw std::invalid_argument(
            "invalid_snapshot: endpoint id fields differ from endpoint JSON");
      }
      std::uint64_t issue_ns = 0;
      if (entry.terminal) {
        if (entry.last_progress.is_null()) {
          throw std::invalid_argument(
              "invalid_snapshot: terminal endpoint lacks retained progress");
        }
        issue_ns = parse_timestamp(entry.last_progress.at("issue_ns"),
                                   "last_progress.issue_ns");
      } else {
        if (!entry.last_progress.is_null() || !entry.handle) {
          throw std::invalid_argument(
              "invalid_snapshot: active endpoint has invalid retained progress");
        }
        const auto runtime_execution = runtime_executions.find(*entry.handle);
        if (runtime_execution == runtime_executions.end()) {
          throw std::invalid_argument(
              "invalid_snapshot: active endpoint handle is absent from Runtime");
        }
        issue_ns = parse_timestamp(runtime_execution->second->at("issue_ns"),
                                   "runtime.execution.issue_ns");
      }
      coarse_die::Runtime validator(hardware);
      const std::uint64_t validation_handle =
          validator.submit_endpoint(entry.endpoint, entry.owner, issue_ns);
      const nlohmann::json expected_progress =
          validator.query_progress(validation_handle);
      const std::string plan_fingerprint =
          expected_progress.at("plan_fingerprint").get<std::string>();
      const std::uint64_t launch_ns =
          parse_timestamp(expected_progress.at("ready_ns"),
                          "endpoint.expected_ready_ns") - issue_ns;
      if (entry.terminal) {
        validate_progress(entry.last_progress, entry.owner,
                          plan_fingerprint, hardware->fingerprint, 0,
                          launch_ns, true, entry.cancelled, entry.finish_ns,
                          now_ns, observed_history_handles);
      } else if (entry.cancelled) {
        throw std::invalid_argument(
            "invalid_snapshot: active endpoint cannot be cancelled");
      }
      if (!entry.terminal) {
        const auto [id_it, id_inserted] = expected_endpoint_ids.emplace(
            entry.endpoint_id, leg_id);
        (void)id_it;
        if (!id_inserted) {
          throw std::invalid_argument(
              "invalid_snapshot: duplicate live endpoint id");
        }
        auto tx = expected_endpoint_transactions.find(entry.transaction_id);
        if (tx == expected_endpoint_transactions.end()) {
          expected_endpoint_transactions.emplace(
              entry.transaction_id,
              EndpointTransactionOwner{entry.owner, 1});
        } else {
          if (!same_owner(tx->second.owner, entry.owner) ||
              tx->second.live_endpoints ==
                  std::numeric_limits<std::size_t>::max()) {
            throw std::invalid_argument(
                "invalid_snapshot: live endpoint transaction owner/refcount is invalid");
          }
          ++tx->second.live_endpoints;
        }
      }
    }
    imported_entries.emplace(leg_id, std::move(entry));
  }

  // Import through the canonical Runtime validator before trusting any live
  // handle. This creates a local object only; a failure cannot affect a
  // source domain or publish partially restored wrapper state.
  auto imported_runtime =
      coarse_die::Runtime::import_snapshot(*hardware, runtime_wire);
  std::set<std::uint64_t> wrapper_live_handles;
  for (auto& [leg_id, entry] : imported_entries) {
    (void)leg_id;
    if (entry.terminal) continue;
    const std::uint64_t handle = *entry.handle;
    if (!wrapper_live_handles.insert(handle).second) {
      throw std::invalid_argument(
          "invalid_snapshot: multiple legs bind the same live handle");
    }
    const auto raw_execution = runtime_executions.find(handle);
    if (raw_execution == runtime_executions.end()) {
      throw std::invalid_argument(
          "invalid_snapshot: wrapper handle is absent from Runtime");
    }
    const auto& wire_execution = *raw_execution->second;
    if (wire_execution.at("terminal").get<bool>() ||
        !same_owner_json(wire_execution.at("owner"), entry.owner)) {
      throw std::invalid_argument(
          "invalid_snapshot: live Runtime handle owner/lifecycle differs from entry");
    }
    const nlohmann::json active_progress =
        imported_runtime->query_progress(handle);
    std::string expected_fingerprint;
    std::uint64_t expected_launch_ns = 0;
    if (entry.kind == EntryKind::PlanChain) {
      const auto& plan = entry.plans[entry.plan_index];
      const nlohmann::json plan_wire = coarse_die::plan_to_json(*plan);
      if (wire_execution.at("plan") != plan_wire) {
        throw std::invalid_argument(
            "invalid_snapshot: active Runtime Plan differs from chain cursor");
      }
      expected_fingerprint = plan->fingerprint;
      expected_launch_ns = plan->launch_ns;
      if (entry.plan_index > 0 &&
          parse_timestamp(active_progress.at("issue_ns"),
                          "active_progress.issue_ns") !=
              parse_timestamp(
                  entry.completed_plan_progress.back().at("finish_ns"),
                  "completed_plan_progress.finish_ns")) {
        throw std::invalid_argument(
            "invalid_snapshot: active Plan did not start at prior terminal tick");
      }
      if (entry.plan_index > 0 &&
          handle <= parse_u64(
                        entry.completed_plan_progress.back().at("handle"),
                        "completed_plan_progress.handle")) {
        throw std::invalid_argument(
            "invalid_snapshot: active Plan handle does not follow chain history");
      }
    } else {
      const auto& task = wire_execution.at("plan").at("task");
      if (!task.is_object() || !task.contains("endpoint") ||
          task.at("endpoint") != entry.endpoint) {
        throw std::invalid_argument(
            "invalid_snapshot: active Runtime endpoint differs from wrapper JSON");
      }
      expected_fingerprint =
          active_progress.at("plan_fingerprint").get<std::string>();
      const std::uint64_t issue_ns = parse_timestamp(
          active_progress.at("issue_ns"), "active_progress.issue_ns");
      const std::uint64_t ready_ns = parse_timestamp(
          active_progress.at("ready_ns"), "active_progress.ready_ns");
      if (ready_ns < issue_ns) {
        throw std::invalid_argument(
            "invalid_snapshot: active endpoint ready time precedes issue");
      }
      expected_launch_ns = ready_ns - issue_ns;
    }
    validate_progress(active_progress, entry.owner, expected_fingerprint,
                      hardware->fingerprint, handle, expected_launch_ns,
                      false, false, std::nullopt, now_ns,
                      observed_history_handles);
  }

  std::set<std::uint64_t> runtime_live_handles;
  for (const std::uint64_t handle : imported_runtime->execution_handles()) {
    const auto finish = imported_runtime->terminal_finish_ns(handle);
    if (finish.has_value() || !runtime_live_handles.insert(handle).second) {
      throw std::invalid_argument(
          "invalid_snapshot: Runtime retains a terminal or duplicate execution handle");
    }
  }
  if (runtime_live_handles != wrapper_live_handles) {
    throw std::invalid_argument(
        "invalid_snapshot: wrapper and Runtime live handles are not a bijection");
  }
  for (const auto handle : observed_history_handles) {
    if (next_handle == 0 || handle >= next_handle) {
      throw std::invalid_argument(
          "invalid_snapshot: retained handle is not below Runtime next_handle");
    }
  }
  if (observed_history_handles.size() < wrapper_live_handles.size()) {
    throw std::invalid_argument(
        "invalid_snapshot: live handle was not included in progress identity checks");
  }
  for (const auto handle : wrapper_live_handles) {
    if (handle >= next_handle ||
        !observed_history_handles.count(handle)) {
      throw std::invalid_argument(
          "invalid_snapshot: live handle is invalid or reused by retained history");
    }
  }

  nlohmann::json expected_endpoint_ids_wire = nlohmann::json::array();
  for (const auto& [endpoint_id, leg_id] : expected_endpoint_ids) {
    expected_endpoint_ids_wire.push_back(
        {{"endpoint_id", endpoint_id}, {"leg_id", leg_id}});
  }
  nlohmann::json expected_transactions_wire = nlohmann::json::array();
  nlohmann::json expected_runtime_transactions = nlohmann::json::array();
  for (const auto& [transaction_id, record] :
       expected_endpoint_transactions) {
    expected_transactions_wire.push_back(
        {{"transaction_id", transaction_id},
         {"owner", owner_json(record.owner)},
         {"live_endpoints", record.live_endpoints}});
    expected_runtime_transactions.push_back(
        {{"transaction_id", transaction_id},
         {"owner", owner_json(record.owner)},
         {"refcount", record.live_endpoints}});
  }
  require_keys(snapshot.at("endpoint_bindings"),
               {"endpoint_ids", "transactions"}, "endpoint_bindings");
  if (!snapshot.at("endpoint_bindings").at("endpoint_ids").is_array() ||
      !snapshot.at("endpoint_bindings").at("transactions").is_array() ||
      snapshot.at("endpoint_bindings").at("endpoint_ids") !=
          expected_endpoint_ids_wire ||
      snapshot.at("endpoint_bindings").at("transactions") !=
          expected_transactions_wire) {
    throw std::invalid_argument(
        "invalid_snapshot: endpoint bindings differ from active wrapper entries");
  }
  nlohmann::json expected_runtime_endpoint_registry = nlohmann::json::array();
  for (const auto& [endpoint_id, leg_id] : expected_endpoint_ids) {
    const auto entry = imported_entries.find(leg_id);
    if (entry == imported_entries.end()) {
      throw std::invalid_argument(
          "invalid_snapshot: endpoint binding references a missing leg");
    }
    expected_runtime_endpoint_registry.push_back(
        {{"endpoint_id", endpoint_id},
         {"transaction_id", entry->second.transaction_id}});
  }
  if (!runtime_wire.at("endpoint_registry").is_array() ||
      !runtime_wire.at("transaction_owners").is_array() ||
      runtime_wire.at("endpoint_registry") !=
          expected_runtime_endpoint_registry ||
      runtime_wire.at("transaction_owners") !=
          expected_runtime_transactions) {
    throw std::invalid_argument(
        "invalid_snapshot: Runtime endpoint registry differs from wrapper bindings");
  }

  std::set<std::string> queued_legs;
  std::vector<NativeCoarseDieCompletion> imported_completions;
  std::optional<std::uint64_t> previous_completion_ns;
  for (const auto& completion_wire : snapshot.at("completion_queue")) {
    require_keys(completion_wire, {"leg_id", "finish_ns", "cancelled"},
                 "completion_queue item");
    if (!completion_wire.at("leg_id").is_string() ||
        !completion_wire.at("cancelled").is_boolean()) {
      throw std::invalid_argument(
          "invalid_snapshot: completion item identity/flags are invalid");
    }
    const std::string leg_id = completion_wire.at("leg_id").get<std::string>();
    const std::uint64_t finish_ns =
        parse_timestamp(completion_wire.at("finish_ns"),
                        "completion.finish_ns");
    const bool cancelled = completion_wire.at("cancelled").get<bool>();
    const auto entry = imported_entries.find(leg_id);
    if (entry == imported_entries.end() || !entry->second.terminal ||
        entry->second.finish_ns != finish_ns ||
        entry->second.cancelled != cancelled ||
        (previous_completion_ns && finish_ns < *previous_completion_ns) ||
        !queued_legs.insert(leg_id).second) {
      throw std::invalid_argument(
          "invalid_snapshot: queued completion is duplicate or differs from terminal entry");
    }
    previous_completion_ns = finish_ns;
    imported_completions.push_back({leg_id, finish_ns, cancelled});
  }

  auto result = std::unique_ptr<NativeCoarseDieDomain>(
      new NativeCoarseDieDomain(std::move(resource_key), std::move(hardware),
                                *imported_runtime));
  result->entries_ = std::move(imported_entries);
  result->active_endpoint_ids_ = std::move(expected_endpoint_ids);
  result->active_endpoint_transactions_ =
      std::move(expected_endpoint_transactions);
  result->completions_ = std::move(imported_completions);
  return result;
}

std::unique_ptr<NativeCoarseDieDomain>
NativeCoarseDieDomain::import_local_sample(
    std::string resource_key,
    std::shared_ptr<const coarse_die::Hardware> hardware,
    std::uint64_t expected_rank, const nlohmann::json& sample,
    const std::map<std::uint64_t, std::string>& handle_to_leg_id) {
  require_keys(sample,
               {"schema", "rank", "sampled_at_ns", "live_runtime_now_ns",
                "hardware_config", "hardware_fingerprint", "runtime_snapshot",
                "completion_bindings", "scope", "callbacks_dispatched",
                "production_world_import_supported"},
               "local sample");
  require_finite_json(sample, "local sample");
  if (resource_key.empty() || !hardware ||
      sample.at("schema") != "coarse_die_local_sample_v1" ||
      parse_u64(sample.at("rank"), "sample.rank") != expected_rank ||
      sample.at("scope") != "rank_local_runtime_only" ||
      !sample.at("callbacks_dispatched").is_boolean() ||
      sample.at("callbacks_dispatched").get<bool>() ||
      !sample.at("production_world_import_supported").is_boolean() ||
      sample.at("production_world_import_supported").get<bool>()) {
    throw std::invalid_argument(
        "invalid_local_sample: schema, rank, scope, or callback flags are incompatible");
  }

  const std::uint64_t sampled_at_ns =
      parse_timestamp(sample.at("sampled_at_ns"), "sample.sampled_at_ns");
  const std::uint64_t live_runtime_now_ns = parse_timestamp(
      sample.at("live_runtime_now_ns"), "sample.live_runtime_now_ns");
  if (live_runtime_now_ns > sampled_at_ns ||
      !sample.at("hardware_fingerprint").is_string() ||
      sample.at("hardware_fingerprint").get<std::string>() !=
          hardware->fingerprint ||
      !sample.at("hardware_config").is_object()) {
    throw std::invalid_argument(
        "incompatible_local_sample: tick or Hardware identity differs");
  }
  const coarse_die::Hardware sampled_hardware =
      coarse_die::resolve_hardware(sample.at("hardware_config"));
  if (sampled_hardware.fingerprint != hardware->fingerprint ||
      sampled_hardware.derived != hardware->derived ||
      sampled_hardware.rates != hardware->rates) {
    throw std::invalid_argument(
        "incompatible_local_sample: hardware_config resources differ");
  }

  const auto& runtime_wire = sample.at("runtime_snapshot");
  if (!runtime_wire.is_object() ||
      runtime_wire.value("schema", std::string()) !=
          "coarse_die_snapshot_v3" ||
      !runtime_wire.contains("now_ns") ||
      parse_timestamp(runtime_wire.at("now_ns"),
                      "sample.runtime_snapshot.now_ns") != sampled_at_ns ||
      runtime_wire.value("hardware_fingerprint", std::string()) !=
          hardware->fingerprint) {
    throw std::invalid_argument(
        "incompatible_local_sample: Runtime tick or Hardware differs from sample");
  }

  // Import the sampled Runtime first so its canonical parser validates the
  // complete raw execution, Plan, Owner, counters, and endpoint registries.
  // This temporary copy is also the authority for the binding state below.
  auto sampled_runtime =
      coarse_die::Runtime::import_snapshot(*hardware, runtime_wire);
  const auto handles = sampled_runtime->execution_handles();
  if (!sample.at("completion_bindings").is_array() ||
      sample.at("completion_bindings").size() != handles.size() ||
      handle_to_leg_id.size() != handles.size()) {
    throw std::invalid_argument(
        "invalid_local_sample: completion and leg bindings must cover every Runtime handle exactly once");
  }

  std::map<std::uint64_t, const nlohmann::json*> runtime_executions;
  for (const auto& execution : runtime_wire.at("executions")) {
    const std::uint64_t handle =
        parse_u64(execution.at("handle"), "runtime.execution.handle");
    if (!runtime_executions.emplace(handle, &execution).second) {
      throw std::invalid_argument(
          "invalid_local_sample: duplicate Runtime execution handle");
    }
  }

  std::map<std::uint64_t, const nlohmann::json*> completion_bindings;
  for (const auto& binding : sample.at("completion_bindings")) {
    require_keys(binding,
                 {"handle", "owner", "state_at_sample", "terminal_at_sample",
                  "finish_ns_at_sample"},
                 "local sample completion binding");
    const std::uint64_t handle =
        parse_u64(binding.at("handle"), "completion_binding.handle");
    if (!completion_bindings.emplace(handle, &binding).second) {
      throw std::invalid_argument(
          "invalid_local_sample: duplicate completion binding handle");
    }
    const auto execution = runtime_executions.find(handle);
    if (execution == runtime_executions.end()) {
      throw std::invalid_argument(
          "invalid_local_sample: completion binding references an unknown Runtime handle");
    }
    const coarse_die::Owner binding_owner = parse_owner(binding.at("owner"));
    const coarse_die::Owner execution_owner =
        parse_owner(execution->second->at("owner"));
    const nlohmann::json active_progress =
        sampled_runtime->query_progress(handle);
    const bool terminal = execution->second->at("terminal").get<bool>();
    const auto& binding_finish = binding.at("finish_ns_at_sample");
    const auto& execution_finish = execution->second->at("finish_ns");
    if (!same_owner(binding_owner, execution_owner) ||
        !binding.at("state_at_sample").is_string() ||
        binding.at("state_at_sample") != active_progress.at("state") ||
        !binding.at("terminal_at_sample").is_boolean() ||
        binding.at("terminal_at_sample").get<bool>() != terminal ||
        (terminal &&
         (binding_finish.is_null() || execution_finish.is_null() ||
          parse_timestamp(binding_finish,
                          "completion_binding.finish_ns_at_sample") !=
              parse_timestamp(execution_finish,
                              "runtime.execution.finish_ns") ||
          active_progress.at("finish_ns") != binding_finish)) ||
        (!terminal && (!binding_finish.is_null() || !execution_finish.is_null()))) {
      throw std::invalid_argument(
          "invalid_local_sample: completion binding Owner or lifecycle differs from Runtime");
    }
  }
  if (completion_bindings.size() != handles.size()) {
    throw std::invalid_argument(
        "invalid_local_sample: completion bindings are not a Runtime-handle bijection");
  }

  std::map<std::string, nlohmann::json> domain_entries;
  std::map<std::uint64_t, std::string> active_endpoint_ids;
  std::map<std::uint64_t, EndpointTransactionOwner>
      active_endpoint_transactions;
  struct PendingSampleCompletion {
    std::uint64_t handle = 0;
    nlohmann::json wire;
  };
  std::vector<PendingSampleCompletion> pending_completions;

  for (const std::uint64_t handle : handles) {
    const auto mapping = handle_to_leg_id.find(handle);
    const auto execution_it = runtime_executions.find(handle);
    const auto binding_it = completion_bindings.find(handle);
    if (mapping == handle_to_leg_id.end() ||
        execution_it == runtime_executions.end() ||
        binding_it == completion_bindings.end()) {
      throw std::invalid_argument(
          "invalid_local_sample: explicit handle-to-leg mapping is incomplete");
    }
    const std::string& leg_id = mapping->second;
    require_leg_id(leg_id);
    if (domain_entries.find(leg_id) != domain_entries.end()) {
      throw std::invalid_argument(
          "invalid_local_sample: explicit leg ids must be unique");
    }

    const auto& raw_execution = *execution_it->second;
    const coarse_die::Owner owner = parse_owner(raw_execution.at("owner"));
    const auto plan = coarse_die::plan_from_json(raw_execution.at("plan"));
    if (!plan || plan->hardware_fingerprint != hardware->fingerprint ||
        coarse_die::plan_to_json(*plan) != raw_execution.at("plan")) {
      throw std::invalid_argument(
          "invalid_local_sample: Runtime Plan is not canonical for this Hardware");
    }
    const bool endpoint_plan =
        plan->task.is_object() &&
        plan->task.value("schema", std::string()) ==
            "coarse_die_endpoint_v2" &&
        plan->task.contains("endpoint");
    const bool terminal = raw_execution.at("terminal").get<bool>();
    const bool cancelled = raw_execution.at("cancelled").get<bool>();
    const nlohmann::json progress = sampled_runtime->query_progress(handle);

    nlohmann::json entry = {
        {"leg_id", leg_id},
        {"kind", endpoint_plan ? "endpoint" : "plan_chain"},
        {"owner", owner_json(owner)},
        {"handle", terminal ? nlohmann::json(nullptr)
                             : nlohmann::json(handle)},
        {"terminal", terminal},
        {"cancelled", cancelled},
        {"finish_ns", terminal ? raw_execution.at("finish_ns")
                                : nlohmann::json(nullptr)},
        {"last_progress", terminal ? progress : nlohmann::json(nullptr)},
        {"plan_index", 0},
        {"completed_plan_progress", nlohmann::json::array()}};

    if (endpoint_plan) {
      const nlohmann::json endpoint = plan->task.at("endpoint");
      const std::uint64_t endpoint_id = parse_u64(
          endpoint.at("endpoint_id"), "endpoint.endpoint_id");
      const std::uint64_t transaction_id = parse_u64(
          endpoint.at("transaction_id"), "endpoint.transaction_id");
      entry["endpoint"] = endpoint;
      entry["endpoint_id"] = endpoint_id;
      entry["transaction_id"] = transaction_id;
      if (!terminal) {
        if (!active_endpoint_ids.emplace(endpoint_id, leg_id).second) {
          throw std::invalid_argument(
              "invalid_local_sample: duplicate active endpoint id");
        }
        auto tx = active_endpoint_transactions.find(transaction_id);
        if (tx == active_endpoint_transactions.end()) {
          active_endpoint_transactions.emplace(
              transaction_id, EndpointTransactionOwner{owner, 1});
        } else if (!same_owner(tx->second.owner, owner) ||
                   tx->second.live_endpoints ==
                       std::numeric_limits<std::size_t>::max()) {
          throw std::invalid_argument(
              "invalid_local_sample: active transaction Owner/refcount is invalid");
        } else {
          ++tx->second.live_endpoints;
        }
      }
    } else {
      entry["plans"] = nlohmann::json::array({raw_execution.at("plan")});
      if (terminal && !cancelled) {
        entry["completed_plan_progress"].push_back(progress);
      }
    }

    if (terminal) {
      pending_completions.push_back(
          {handle,
           {{"leg_id", leg_id},
            {"finish_ns", raw_execution.at("finish_ns")},
            {"cancelled", cancelled}}});
      // Reclaim only this private imported Runtime. Runtime owns the endpoint
      // registry/refcount cleanup, while the copied progress above remains in
      // the wrapper entry and the completion stays pending for the caller.
      sampled_runtime->destroy_execution(handle);
    }
    domain_entries.emplace(leg_id, std::move(entry));
  }

  std::sort(pending_completions.begin(), pending_completions.end(),
            [](const PendingSampleCompletion& left,
               const PendingSampleCompletion& right) {
              const auto left_finish =
                  left.wire.at("finish_ns").get<std::uint64_t>();
              const auto right_finish =
                  right.wire.at("finish_ns").get<std::uint64_t>();
              if (left_finish != right_finish) return left_finish < right_finish;
              // Runtime::execution_handles() is in ascending handle order,
              // matching the Sys callback ready-vector order for tied ticks.
              return left.handle < right.handle;
            });
  nlohmann::json completion_queue = nlohmann::json::array();
  for (const auto& completion : pending_completions) {
    completion_queue.push_back(completion.wire);
  }

  nlohmann::json endpoint_ids_wire = nlohmann::json::array();
  for (const auto& [endpoint_id, leg_id] : active_endpoint_ids) {
    endpoint_ids_wire.push_back(
        {{"endpoint_id", endpoint_id}, {"leg_id", leg_id}});
  }
  nlohmann::json transactions_wire = nlohmann::json::array();
  nlohmann::json runtime_transactions_wire = nlohmann::json::array();
  for (const auto& [transaction_id, record] : active_endpoint_transactions) {
    transactions_wire.push_back(
        {{"transaction_id", transaction_id},
         {"owner", owner_json(record.owner)},
         {"live_endpoints", record.live_endpoints}});
    runtime_transactions_wire.push_back(
        {{"transaction_id", transaction_id},
         {"owner", owner_json(record.owner)},
         {"refcount", record.live_endpoints}});
  }
  nlohmann::json entries_wire = nlohmann::json::array();
  for (auto& [leg_id, entry] : domain_entries) {
    (void)leg_id;
    entries_wire.push_back(std::move(entry));
  }

  const nlohmann::json filtered_runtime = sampled_runtime->export_snapshot();
  const nlohmann::json domain_snapshot = {
      {"schema", kDomainSnapshotSchema},
      {"snapshot_version", 1},
      {"resource_key", resource_key},
      {"hardware_fingerprint", hardware->fingerprint},
      {"runtime", filtered_runtime},
      {"entries", std::move(entries_wire)},
      {"endpoint_bindings",
       {{"endpoint_ids", std::move(endpoint_ids_wire)},
        {"transactions", std::move(transactions_wire)}}},
      {"completion_queue", std::move(completion_queue)}};
  return import_snapshot(std::move(resource_key), std::move(hardware),
                         domain_snapshot);
}

nlohmann::json NativeCoarseDieDomain::export_runtime_snapshot() const {
  return runtime_.export_snapshot();
}

std::unique_ptr<NativeCoarseDieDomain>
NativeCoarseDieDomain::fork_snapshot(const NativeCoarseDieDomain& source) {
  auto fork = std::unique_ptr<NativeCoarseDieDomain>(
      new NativeCoarseDieDomain(source.resource_key_, source.hardware_,
                                source.runtime_));
  fork->entries_ = source.entries_;
  fork->active_endpoint_ids_ = source.active_endpoint_ids_;
  fork->active_endpoint_transactions_ =
      source.active_endpoint_transactions_;
  fork->completions_ = source.completions_;
  return fork;
}

}  // namespace AstraSim::ExecutionDriven
