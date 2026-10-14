#include "astra-sim/workload/execution_driven/NativeCoarseDieDomain.hh"

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using AstraSim::ExecutionDriven::NativeCoarseDieDomain;
using AstraSim::ExecutionDriven::NativeCoarseDieCompletion;
using coarse_die::Activity;
using coarse_die::Epoch;
using coarse_die::Hardware;
using coarse_die::Owner;
using coarse_die::Plan;
using coarse_die::Role;
using nlohmann::json;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

Hardware test_hardware(double hbm0_rate = 1.0e9,
                       double compute_rate = 0.0) {
  std::ifstream input(COARSE_DIE_HW_CONFIG, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open canonical hardware fixture");
  const json config = json::parse(std::string(
      std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()));
  Hardware hardware = coarse_die::resolve_hardware(config);
  hardware.rates[coarse_die::COMPUTE_GATE] = compute_rate;
  hardware.rates[coarse_die::HBM0] = hbm0_rate;
  static const std::array<const char*, coarse_die::resource_count> ids = {
      "COMPUTE_GATE", "SRAM_READ", "SRAM_WRITE", "LOCAL", "HBM0",
      "HBM1", "INJECT0", "INJECT1", "XPOS", "XNEG", "YPOS", "YNEG",
      "HOP"};
  auto& resources = hardware.derived.at("resources");
  for (std::size_t i = 0; i < hardware.rates.size(); ++i) {
    resources[i]["resource_id"] = i;
    resources[i]["id"] = ids[i];
    resources[i]["rate"] = hardware.rates[i];
  }
  return hardware;
}

Owner test_owner(std::uint64_t id) {
  Owner owner;
  owner.request_id = id;
  owner.node_id = id + 10;
  owner.execution_id = id + 100;
  return owner;
}

std::shared_ptr<const Plan> hbm_plan(const Hardware& hardware,
                                     std::uint64_t id,
                                     std::uint64_t bytes,
                                     const std::string& name) {
  Activity activity;
  activity.id = id;
  activity.role = Role::Load;
  activity.work[coarse_die::HBM0] = static_cast<double>(bytes);
  activity.hbm_bytes[0][0] = bytes;
  activity.canonical_q = static_cast<double>(bytes);
  Epoch epoch;
  epoch.name = name;
  epoch.activities.push_back(activity);
  auto plan = std::make_shared<Plan>();
  plan->task = {{"name", name}};
  plan->summary = {{"status", "native_coarse_domain_test"}};
  plan->hardware_fingerprint = hardware.fingerprint;
  plan->epochs.push_back(std::move(epoch));
  plan->fingerprint = coarse_die::plan_to_json(*plan)
                          .at("fingerprint").get<std::string>();
  return plan;
}

void chain_successor_starts_at_prior_terminal_tick() {
  const Hardware hardware = test_hardware();
  NativeCoarseDieDomain domain("hbm:0", hardware);
  domain.begin_plan_chain("chain", {hbm_plan(hardware, 1, 3, "first"),
                                     hbm_plan(hardware, 2, 2, "second")},
                          test_owner(1), 0);
  require(domain.next_event() == std::optional<std::uint64_t>(3),
          "domain exposes the first native Runtime event");

  // A single large outer advance must still submit Plan 2 at Plan 1's native
  // terminal tick instead of waiting until the requested target.
  domain.advance_to(100);
  const json progress = domain.progress("chain");
  require(progress.at("terminal").get<bool>(),
          "the entire chain completes during the outer advance");
  require(progress.at("finish_ns").get<std::uint64_t>() == 5,
          "chained Plans consume their exact aggregate service time");
  require(progress.at("completed_plan_progress").size() == 2,
          "each terminal Plan progress survives Runtime handle retirement");
  const json& first = progress.at("completed_plan_progress").at(0);
  const json& second = progress.at("completed_plan_progress").at(1);
  require(first.at("issue_ns").get<std::uint64_t>() == 0 &&
              first.at("finish_ns").get<std::uint64_t>() == 3,
          "first Plan starts at arrival and finishes at 3 ns");
  require(second.at("issue_ns").get<std::uint64_t>() == 3 &&
              second.at("finish_ns").get<std::uint64_t>() == 5,
          "successor is submitted at the actual preceding terminal tick");
  require(!domain.retire("chain"),
          "retirement waits until the corresponding completion is consumed");
  require(domain.progress("chain").at("finish_ns").get<std::uint64_t>() == 5,
          "pending completion keeps its original leg binding intact");
  const auto completions = domain.take_completions();
  require(completions.size() == 1 && completions.front().leg_id == "chain" &&
              completions.front().finish_ns == 5 &&
              !completions.front().cancelled,
          "one external terminal is emitted for the whole chain");
  require(domain.take_completions().empty(),
          "completion delivery is one-shot");
  require(domain.retire("chain"),
          "terminal progress can be retired once the bridge is finished");
  require(!domain.retire("chain"), "retirement is idempotent");
}

void cancellation_keeps_served_work_and_emits_one_terminal() {
  const Hardware hardware = test_hardware();
  NativeCoarseDieDomain domain("hbm:0", hardware);
  domain.begin_plan_chain("cancel", {hbm_plan(hardware, 3, 10, "cancelled")},
                          test_owner(2), 0);
  domain.advance_to(4);
  require(domain.cancel("cancel", 4), "live leg is cancelled");
  require(!domain.cancel("cancel", 4), "terminal leg cannot be cancelled twice");
  const json progress = domain.progress("cancel");
  require(progress.at("state") == "cancelled" &&
              progress.at("finish_ns").get<std::uint64_t>() == 4,
          "cancel progress retains the actual cancellation timestamp");
  require(progress.at("last_progress").at("cancelled").get<bool>(),
          "the preserved native Runtime progress records cancellation");
  require(progress.at("completed_plan_progress").empty(),
          "a cancelled partial Plan is not labeled as completed work");
  const json counters = domain.counters();
  const double served = counters.at("hbm_served_bytes_by_row_direction")
                            .at(0).at(0).get<double>();
  require(std::abs(served - 4.0) < 1e-9,
          "cancellation preserves only the four bytes served before cancellation");
  const auto completions = domain.take_completions();
  require(completions.size() == 1 && completions.front().cancelled &&
              completions.front().finish_ns == 4,
          "cancellation emits exactly one external terminal");
  require(domain.take_completions().empty(),
          "cancelled completion is not emitted again");
  require(domain.retire("cancel"),
          "cancelled terminal state retires after completion consumption");
}

void invalid_later_plan_is_rejected_before_domain_advance() {
  const Hardware hardware = test_hardware();
  NativeCoarseDieDomain domain("hbm:0", hardware);
  domain.begin_plan_chain("blocker", {hbm_plan(hardware, 30, 20, "blocker")},
                          test_owner(30), 0);

  auto invalid_later = std::make_shared<Plan>(
      *hbm_plan(hardware, 32, 1, "wrong_hardware"));
  invalid_later->hardware_fingerprint = "not-the-domain-hardware";
  invalid_later->fingerprint.clear();
  invalid_later->fingerprint = coarse_die::plan_to_json(*invalid_later)
                                   .at("fingerprint").get<std::string>();
  bool rejected = false;
  try {
    domain.begin_plan_chain(
        "invalid-chain",
        {hbm_plan(hardware, 31, 1, "valid_first"), invalid_later},
        test_owner(31), 5);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "hardware-incompatible later Plan is rejected");
  const json blocker = domain.progress("blocker");
  require(blocker.at("runtime_now_ns").get<std::uint64_t>() == 0 &&
              blocker.at("active_execution").at("elapsed_ns").get<std::uint64_t>() == 0,
          "preflight rejects a later Plan before advancing existing work");
  require(domain.counters().at("hbm_served_bytes_by_row_direction")
                  .at(0).at(0).get<double>() == 0.0,
          "failed chain validation does not charge domain service");
  require(domain.export_runtime_snapshot().at("now_ns").get<std::uint64_t>() == 0,
          "failed chain validation leaves the Runtime clock unchanged");
}

void invalid_endpoint_is_rejected_before_domain_advance() {
  const Hardware hardware = test_hardware();
  NativeCoarseDieDomain domain("hbm:0", hardware);
  domain.begin_plan_chain("blocker", {hbm_plan(hardware, 40, 20, "blocker")},
                          test_owner(40), 0);
  const json malformed_endpoint{
      {"endpoint_id", 400}, {"transaction_id", 400}, {"owner_kind", "comm"},
      {"direction", "read"}, {"physical_bytes", 2},
      {"path_class", "hbm_to_edge"}, {"row_fraction", {1.0, 0.0}}};
  bool rejected = false;
  try {
    domain.begin_endpoint("bad-endpoint", malformed_endpoint,
                          test_owner(41), 5);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "malformed endpoint is rejected by private Runtime preflight");
  require(domain.progress("blocker").at("runtime_now_ns").get<std::uint64_t>() == 0 &&
              domain.counters().at("hbm_served_bytes_by_row_direction")
                      .at(0).at(0).get<double>() == 0.0,
          "endpoint validation failure does not advance existing work");
}

void endpoint_completion_releases_native_endpoint_binding() {
  const Hardware hardware = test_hardware();
  NativeCoarseDieDomain domain("hbm:0", hardware);
  const json endpoint{
      {"endpoint_id", 700}, {"transaction_id", 700},
      {"owner_kind", "comm"}, {"direction", "read"},
      {"physical_bytes", 4}, {"row_fraction", {1.0, 0.0}},
      {"path_class", "hbm_to_edge"}, {"eta_hbm", 1.0},
      {"route", {{"internal_route", false}}}};
  const Owner owner = test_owner(70);
  domain.begin_endpoint("endpoint-1", endpoint, owner, 0);
  domain.advance_to(5);
  require(domain.progress("endpoint-1").at("finish_ns").get<std::uint64_t>() == 4,
          "endpoint retires at the native Runtime's finish tick");
  const auto first_completion = domain.take_completions();
  require(first_completion.size() == 1 && first_completion[0].finish_ns == 4,
          "endpoint has one leg-level completion");

  domain.begin_endpoint("endpoint-2", endpoint, owner, 5);
  domain.advance_to(10);
  require(domain.progress("endpoint-2").at("finish_ns").get<std::uint64_t>() == 9,
          "terminal endpoint registry is released for later reuse");
  const auto second_completion = domain.take_completions();
  require(second_completion.size() == 1 &&
              second_completion[0].leg_id == "endpoint-2" &&
              second_completion[0].finish_ns == 9,
          "reused endpoint identity emits only the new leg terminal");
}

void same_tick_chain_dispatch_is_stable_across_registration_order() {
  const Hardware hardware = test_hardware();
  auto run = [&](bool reverse) {
    NativeCoarseDieDomain domain("hbm:0", hardware);
    auto begin_a = [&] {
      domain.begin_plan_chain(
          "a", {hbm_plan(hardware, 51, 1, "a0"),
                hbm_plan(hardware, 52, 2, "a1")},
          test_owner(51), 0);
    };
    auto begin_z = [&] {
      domain.begin_plan_chain(
          "z", {hbm_plan(hardware, 61, 1, "z0"),
                hbm_plan(hardware, 62, 2, "z1")},
          test_owner(61), 0);
    };
    if (reverse) {
      begin_z();
      begin_a();
    } else {
      begin_a();
      begin_z();
    }
    domain.advance_to(10);
    json summary = json::object();
    for (const std::string leg : {"a", "z"}) {
      const json progress = domain.progress(leg);
      const auto& first = progress.at("completed_plan_progress").at(0);
      const auto& second = progress.at("completed_plan_progress").at(1);
      summary[leg] = {{"first_finish_ns", first.at("finish_ns")},
                      {"second_issue_ns", second.at("issue_ns")},
                      {"second_finish_ns", second.at("finish_ns")}};
    }
    const auto completions = domain.take_completions();
    require(completions.size() == 2 && completions[0].leg_id == "a" &&
                completions[1].leg_id == "z",
            "same-tick completion order follows stable leg-id order");
    return summary;
  };
  const json forward = run(false);
  const json reverse = run(true);
  require(forward == reverse,
          "same-tick successor service is independent of registration order");
  require(forward.at("a").at("first_finish_ns").get<std::uint64_t>() == 2 &&
              forward.at("z").at("first_finish_ns").get<std::uint64_t>() == 2 &&
              forward.at("a").at("second_issue_ns").get<std::uint64_t>() == 2 &&
              forward.at("z").at("second_issue_ns").get<std::uint64_t>() == 2,
          "both same-tick successors are submitted at the shared 2 ns terminal");
}

void exact_json_nanosecond_boundary_is_enforced() {
  const Hardware hardware = test_hardware();
  NativeCoarseDieDomain domain("hbm:0", hardware);
  bool rejected = false;
  try {
    domain.advance_to(9007199254740992ULL);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "nanoseconds above 2^53-1 are rejected");
  require(domain.export_runtime_snapshot().at("now_ns").get<std::uint64_t>() == 0,
          "out-of-range time rejection leaves Runtime untouched");
}

void fork_copies_runtime_and_chain_bindings_without_aliasing() {
  const Hardware hardware = test_hardware();
  NativeCoarseDieDomain domain("hbm:0", hardware);
  domain.begin_plan_chain("forked", {hbm_plan(hardware, 4, 10, "forked")},
                          test_owner(3), 0);
  domain.advance_to(2);
  auto fork = NativeCoarseDieDomain::fork_snapshot(domain);

  fork->advance_to(5);
  require(domain.progress("forked").at("runtime_now_ns").get<std::uint64_t>() == 2,
          "advancing a fork does not advance the source Runtime");
  require(fork->progress("forked").at("runtime_now_ns").get<std::uint64_t>() == 5,
          "fork keeps an independent copy of the active chain binding");
  require(fork->cancel("forked", 5),
          "fork can cancel its copied live Runtime handle");
  require(domain.progress("forked").at("state") == "running",
          "fork cancellation does not terminalize the source leg");
  const auto source_completions = domain.take_completions();
  const auto fork_completions = fork->take_completions();
  require(source_completions.empty() && fork_completions.size() == 1 &&
              fork_completions.front().cancelled,
          "fork completion queue is copied but independently mutable");

  const json runtime_snapshot = domain.export_runtime_snapshot();
  require(runtime_snapshot.at("schema") == "coarse_die_snapshot_v3",
          "runtime export uses canonical coarse_die snapshot wire format");
}

}  // namespace

int main() {
  try {
    chain_successor_starts_at_prior_terminal_tick();
    cancellation_keeps_served_work_and_emits_one_terminal();
    invalid_later_plan_is_rejected_before_domain_advance();
    invalid_endpoint_is_rejected_before_domain_advance();
    endpoint_completion_releases_native_endpoint_binding();
    same_tick_chain_dispatch_is_stable_across_registration_order();
    exact_json_nanosecond_boundary_is_enforced();
    fork_copies_runtime_and_chain_bindings_without_aliasing();
    std::cout << "PASS native_coarse_die_domain_test\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL native_coarse_die_domain_test: " << error.what() << '\n';
    return 1;
  }
}
