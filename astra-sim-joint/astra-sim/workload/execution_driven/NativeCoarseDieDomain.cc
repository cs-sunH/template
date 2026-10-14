#include "NativeCoarseDieDomain.hh"

#include <limits>
#include <stdexcept>
#include <utility>

namespace AstraSim::ExecutionDriven {
namespace {

constexpr std::uint64_t kMaxExactJsonNs = 9007199254740991ULL;

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
