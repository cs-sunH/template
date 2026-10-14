#pragma once

#include "coarse_die/Execution.hh"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace AstraSim::ExecutionDriven {

struct NativeCoarseDieCompletion {
  std::string leg_id;
  std::uint64_t finish_ns = 0;
  bool cancelled = false;
};

// One shared coarse_die::Runtime for a native HBM physical domain. Domain
// snapshots carry wrapper bindings as well as Runtime state.
class NativeCoarseDieDomain {
 public:
  NativeCoarseDieDomain(std::string resource_key,
                        coarse_die::Hardware hardware);
  NativeCoarseDieDomain(std::string resource_key,
                        std::shared_ptr<const coarse_die::Hardware> hardware);

  const std::string& resource_key() const noexcept { return resource_key_; }
  const coarse_die::Hardware& hardware() const noexcept { return *hardware_; }

  void begin_plan_chain(
      std::string leg_id,
      std::vector<std::shared_ptr<const coarse_die::Plan>> plans,
      coarse_die::Owner owner, std::uint64_t now_ns);
  void begin_endpoint(std::string leg_id, const nlohmann::json& endpoint,
                      coarse_die::Owner owner, std::uint64_t now_ns);

  void advance_to(std::uint64_t absolute_ns);
  std::optional<std::uint64_t> next_event() const;
  std::vector<NativeCoarseDieCompletion> take_completions();
  bool cancel(const std::string& leg_id, std::uint64_t now_ns);

  // Retains terminal progress until explicitly retired so a bridge can
  // inspect it after consuming the one-shot completion record.
  nlohmann::json progress(const std::string& leg_id) const;
  bool retire(const std::string& leg_id);

  nlohmann::json counters() const;
  nlohmann::json export_snapshot() const;
  static std::unique_ptr<NativeCoarseDieDomain> import_snapshot(
      std::string resource_key,
      std::shared_ptr<const coarse_die::Hardware> hardware,
      const nlohmann::json& snapshot);
  // Imports a read-only rank-local Runtime sample. The caller supplies the
  // external leg identity for every sampled handle; this does not restore
  // graph, lease, network, pool, or callback state.
  static std::unique_ptr<NativeCoarseDieDomain> import_local_sample(
      std::string resource_key,
      std::shared_ptr<const coarse_die::Hardware> hardware,
      std::uint64_t expected_rank, const nlohmann::json& sample,
      const std::map<std::uint64_t, std::string>& handle_to_leg_id);
  // Raw Runtime-only diagnostic export. Use export_snapshot() when wrapper
  // chain and endpoint bindings must be preserved on the wire.
  nlohmann::json export_runtime_snapshot() const;
  static std::unique_ptr<NativeCoarseDieDomain> fork_snapshot(
      const NativeCoarseDieDomain& source);

 private:
  enum class EntryKind { PlanChain, Endpoint };
  struct Entry {
    EntryKind kind = EntryKind::PlanChain;
    coarse_die::Owner owner;
    std::vector<std::shared_ptr<const coarse_die::Plan>> plans;
    nlohmann::json endpoint;
    std::uint64_t endpoint_id = 0;
    std::uint64_t transaction_id = 0;
    std::size_t plan_index = 0;
    std::optional<std::uint64_t> handle;
    bool terminal = false;
    bool cancelled = false;
    std::uint64_t finish_ns = 0;
    std::vector<nlohmann::json> completed_plan_progress;
    nlohmann::json last_progress;
  };

  NativeCoarseDieDomain(std::string resource_key,
                        std::shared_ptr<const coarse_die::Hardware> hardware,
                        const coarse_die::Runtime& runtime);
  void validate_arrival(std::uint64_t now_ns) const;
  void submit_current(const std::string& leg_id, Entry& entry,
                      std::uint64_t now_ns);
  void settle_terminals();
  void release_endpoint_binding(const Entry& entry);
  void record_terminal(const std::string& leg_id, Entry& entry,
                       std::uint64_t finish_ns, bool cancelled);

  std::string resource_key_;
  std::shared_ptr<const coarse_die::Hardware> hardware_;
  coarse_die::Runtime runtime_;
  std::map<std::string, Entry> entries_;
  std::map<std::uint64_t, std::string> active_endpoint_ids_;
  struct EndpointTransactionOwner {
    coarse_die::Owner owner;
    std::size_t live_endpoints = 0;
  };
  std::map<std::uint64_t, EndpointTransactionOwner>
      active_endpoint_transactions_;
  std::vector<NativeCoarseDieCompletion> completions_;
};

}  // namespace AstraSim::ExecutionDriven
