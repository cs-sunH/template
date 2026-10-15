#include "astra-sim/workload/execution_driven/NativeCoarseDieDomain.hh"

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
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

json test_hardware_config() {
  std::ifstream input(COARSE_DIE_HW_CONFIG, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open canonical hardware fixture");
  return json::parse(std::string(
      std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()));
}

Hardware test_hardware(double hbm0_rate = 1.0e9,
                       double compute_rate = 0.0) {
  const json config = test_hardware_config();
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

Hardware canonical_test_hardware() {
  return coarse_die::resolve_hardware(test_hardware_config());
}

Owner test_owner(std::uint64_t id) {
  Owner owner;
  owner.request_id = id;
  owner.node_id = id + 10;
  owner.execution_id = id + 100;
  return owner;
}

json owner_wire(const Owner& owner) {
  return {{"request_id", owner.request_id},
          {"node_id", owner.node_id},
          {"execution_id", owner.execution_id},
          {"generation", owner.generation},
          {"key", owner.key()}};
}

std::unique_ptr<NativeCoarseDieDomain> import_domain(
    const json& snapshot, const Hardware& hardware,
    const std::string& resource_key = "hbm:0") {
  return NativeCoarseDieDomain::import_snapshot(
      resource_key, std::make_shared<const Hardware>(hardware), snapshot);
}

void require_bad_domain_snapshot(const json& snapshot,
                                 const Hardware& hardware,
                                 const std::string& message) {
  bool rejected = false;
  try {
    (void)import_domain(snapshot, hardware);
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, message);
}

json local_sample(const Hardware& hardware, const json& runtime_snapshot,
                  std::uint64_t live_runtime_now_ns) {
  auto runtime = coarse_die::Runtime::import_snapshot(hardware,
                                                       runtime_snapshot);
  json bindings = json::array();
  for (const std::uint64_t handle : runtime->execution_handles()) {
    const json progress = runtime->query_progress(handle);
    bindings.push_back(
        {{"handle", handle},
         {"owner", progress.at("owner")},
         {"state_at_sample", progress.at("state")},
         {"terminal_at_sample", progress.at("terminal")},
         {"finish_ns_at_sample", progress.at("finish_ns")}});
  }
  return {{"schema", "coarse_die_local_sample_v1"},
          {"rank", 0},
          {"sampled_at_ns", runtime_snapshot.at("now_ns")},
          {"live_runtime_now_ns", live_runtime_now_ns},
          {"hardware_config", test_hardware_config()},
          {"hardware_fingerprint", hardware.fingerprint},
          {"runtime_snapshot", runtime_snapshot},
          {"completion_bindings", std::move(bindings)},
          {"scope", "rank_local_runtime_only"},
          {"callbacks_dispatched", false},
          {"production_world_import_supported", false}};
}

void require_bad_local_sample(
    const json& sample, const Hardware& hardware,
    const std::map<std::uint64_t, std::string>& handle_to_leg_id,
    std::uint64_t expected_rank, const std::string& message) {
  const json before = sample;
  bool rejected = false;
  try {
    (void)NativeCoarseDieDomain::import_local_sample(
        "hbm:0", std::make_shared<const Hardware>(hardware), expected_rank,
        sample, handle_to_leg_id);
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, message);
  require(sample == before, "failed local-sample import leaves input unchanged");
}

void require_capacity_update_rejected_without_change(
    NativeCoarseDieDomain& domain, std::size_t resource, double rate,
    std::uint64_t now_ns, const std::string& message) {
  const json before = domain.export_snapshot();
  bool rejected = false;
  try {
    domain.update_capacity(resource, rate, now_ns);
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, message);
  require(domain.export_snapshot() == before,
          message + ": invalid capacity event leaves the live Domain unchanged");
}

std::string hbm0_read_served_exact(const json& counters) {
  return counters.at("hbm_served_bytes_by_row_direction_exact")
      .at(0).at(0).get<std::string>();
}

json served_accounting(const json& counters) {
  return {{"hbm_served_bytes_by_row_direction_exact",
           counters.at("hbm_served_bytes_by_row_direction_exact")},
          {"served_by_category", counters.at("served_by_category")}};
}

json endpoint_spec(std::uint64_t endpoint_id,
                   std::uint64_t transaction_id,
                   std::uint64_t bytes = 10) {
  return {{"endpoint_id", endpoint_id},
          {"transaction_id", transaction_id},
          {"owner_kind", "comm"},
          {"direction", "read"},
          {"physical_bytes", bytes},
          {"row_fraction", {1.0, 0.0}},
          {"path_class", "hbm_to_edge"},
          {"eta_hbm", 1.0},
          {"route", {{"internal_route", false}}}};
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

void capacity_updates_reject_invalid_events_before_live_advance() {
  const Hardware hardware = test_hardware();
  NativeCoarseDieDomain domain("hbm:0", hardware);
  domain.begin_plan_chain("capacity-validation-comp",
                          {hbm_plan(hardware, 71, 1000, "capacity-validation")},
                          test_owner(71), 0);
  domain.begin_endpoint("capacity-validation-endpoint",
                        endpoint_spec(710, 710, 1000), test_owner(72), 0);
  domain.advance_to(5);
  const json before = domain.export_snapshot();
  require(before.at("runtime").at("now_ns") == 5 &&
              before.at("runtime").at("executions").size() == 2,
          "invalid-event fixture has active COMP and endpoint work at tick 5");

  require_capacity_update_rejected_without_change(
      domain, coarse_die::resource_count, 0.0, 6,
      "out-of-range capacity resource is rejected");
  require_capacity_update_rejected_without_change(
      domain, coarse_die::HBM0, -1.0, 6,
      "negative physical capacity is rejected");
  require_capacity_update_rejected_without_change(
      domain, coarse_die::HBM0, std::numeric_limits<double>::quiet_NaN(), 6,
      "NaN physical capacity is rejected");
  require_capacity_update_rejected_without_change(
      domain, coarse_die::HBM0,
      std::numeric_limits<double>::infinity(), 6,
      "infinite physical capacity is rejected");
  require_capacity_update_rejected_without_change(
      domain, coarse_die::HBM0, hardware.rates[coarse_die::HBM0] + 1.0, 6,
      "rate above the resolved hardware peak is rejected");
  require_capacity_update_rejected_without_change(
      domain, coarse_die::INJECT0, 1.0, 6,
      "a disabled physical resource cannot be enabled");
  require_capacity_update_rejected_without_change(
      domain, coarse_die::HBM0, 0.5e9, 4,
      "capacity event before the live Runtime clock is rejected");
  require_capacity_update_rejected_without_change(
      domain, coarse_die::HBM0, 0.5e9, 9007199254740992ULL,
      "capacity event outside exact JSON nanosecond range is rejected");
  require(domain.export_snapshot() == before,
          "all invalid capacity events preserve active progress and bindings");

  domain.update_capacity(coarse_die::INJECT0, 0.0, 5);
  require(domain.export_runtime_snapshot().at("capacities")
                  .at(coarse_die::INJECT0) == 0.0,
          "a disabled resource remains valid at zero physical rate");
}

void capacity_updates_retime_shared_comp_and_endpoint_and_survive_forks() {
  const Hardware hardware = test_hardware();
  NativeCoarseDieDomain source("hbm:0", hardware);
  source.begin_plan_chain("capacity-comp",
                          {hbm_plan(hardware, 73, 100, "capacity-comp")},
                          test_owner(73), 0);
  source.begin_endpoint("capacity-endpoint", endpoint_spec(730, 730, 100),
                        test_owner(74), 0);

  source.advance_to(20);
  const json counters_at_20 = source.counters();
  const long double first_service =
      std::stold(hbm0_read_served_exact(counters_at_20));
  require(first_service > 0.0L,
          "COMP and endpoint have both begun serving shared HBM work");

  source.update_capacity(coarse_die::HBM0, 0.5e9, 20);
  require(source.export_runtime_snapshot().at("capacities")
                  .at(coarse_die::HBM0) == 0.5e9,
          "capacity event installs the requested absolute physical rate");
  require(served_accounting(source.counters()) ==
              served_accounting(counters_at_20),
          "capacity update does not rewrite work served before its event tick");
  source.advance_to(40);
  const long double service_at_40 =
      std::stold(hbm0_read_served_exact(source.counters()));
  const long double slower_service = service_at_40 - first_service;
  require(slower_service > 0.0L && slower_service < first_service &&
              std::fabs(slower_service * 2.0L - first_service) < 1.0e-9L,
          "half physical capacity halves aggregate shared-HBM service over equal intervals");

  const json counters_at_40 = source.counters();
  source.update_capacity(coarse_die::HBM0, 0.0, 40);
  require(served_accounting(source.counters()) ==
              served_accounting(counters_at_40),
          "pausing capacity preserves all previously served accounting");
  source.advance_to(50);
  require(hbm0_read_served_exact(source.counters()) ==
              hbm0_read_served_exact(counters_at_40),
          "zero physical rate pauses COMP and endpoint service");
  require(source.progress("capacity-comp").at("state") == "running" &&
              source.progress("capacity-endpoint").at("state") == "running",
          "paused work remains live in both wrapper bindings");

  const json wire = json::parse(source.export_snapshot().dump());
  auto imported = import_domain(wire, hardware);
  auto forked = NativeCoarseDieDomain::fork_snapshot(source);
  const auto require_same_copies = [&](const std::string& stage) {
    const json expected = source.export_snapshot();
    require(imported->export_snapshot() == expected &&
                forked->export_snapshot() == expected,
            "imported and forked Domains match after " + stage);
  };
  const auto update_all = [&](double rate, std::uint64_t tick,
                              const std::string& stage) {
    source.update_capacity(coarse_die::HBM0, rate, tick);
    imported->update_capacity(coarse_die::HBM0, rate, tick);
    forked->update_capacity(coarse_die::HBM0, rate, tick);
    require_same_copies(stage);
  };
  const auto advance_all = [&](std::uint64_t tick,
                               const std::string& stage) {
    source.advance_to(tick);
    imported->advance_to(tick);
    forked->advance_to(tick);
    require_same_copies(stage);
  };
  update_all(1.0e9, 50, "restoring the original physical rate");
  advance_all(70, "the resumed interval");
  update_all(0.25e9, 70, "a second slowdown");
  advance_all(80, "the second slowed interval");
  update_all(0.0, 80, "a second pause");
  advance_all(85, "the second paused interval");
  update_all(1.0e9, 85, "the final capacity restoration");
  advance_all(1000, "completion under the known capacity sequence");
  require(source.progress("capacity-comp").at("terminal").get<bool>() &&
              source.progress("capacity-endpoint").at("terminal").get<bool>() &&
              source.export_snapshot().at("completion_queue").size() == 2,
          "restored physical capacity completes both shared-HBM legs once");
  const auto source_completions = source.take_completions();
  const auto imported_completions = imported->take_completions();
  const auto forked_completions = forked->take_completions();
  require(source_completions.size() == 2 && imported_completions.size() == 2 &&
              forked_completions.size() == 2,
          "capacity-sequence copies each produce one completion per leg");
  require(imported->export_snapshot() == source.export_snapshot() &&
              forked->export_snapshot() == source.export_snapshot(),
          "wire import and fork preserve identical terminal accounting and consumption");
}

void capacity_event_at_chain_terminal_uses_old_rate_then_updates_successor() {
  const Hardware hardware = test_hardware();
  NativeCoarseDieDomain domain("hbm:0", hardware);
  domain.begin_plan_chain("capacity-boundary",
                          {hbm_plan(hardware, 75, 5, "boundary-first"),
                           hbm_plan(hardware, 76, 5, "boundary-second")},
                          test_owner(75), 0);
  require(domain.next_event() == std::optional<std::uint64_t>(5),
          "first chain Plan reaches its terminal boundary at 5 ns");

  domain.update_capacity(coarse_die::HBM0, 0.5e9, 5);
  const json at_boundary = domain.progress("capacity-boundary");
  require(at_boundary.at("plan_index") == 1 &&
              at_boundary.at("completed_plan_progress").size() == 1 &&
              at_boundary.at("completed_plan_progress").at(0).at("finish_ns") ==
                  5 &&
              at_boundary.at("active_execution").at("issue_ns") == 5 &&
              at_boundary.at("active_execution").at("activities").at(0)
                      .at("progress").get<double>() == 0.0,
          "terminal Plan and successor issue are settled before same-tick capacity change");
  require(domain.export_runtime_snapshot().at("capacities")
                  .at(coarse_die::HBM0) == 0.5e9,
          "successor observes the new capacity from its issue tick onward");

  domain.advance_to(15);
  const auto completion = domain.take_completions();
  require(completion.size() == 1 &&
              completion.front().leg_id == "capacity-boundary" &&
              completion.front().finish_ns == 15 &&
              !completion.front().cancelled,
          "successor service uses the reduced rate for exactly ten nanoseconds");
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

void wire_snapshot_roundtrips_fractional_live_progress() {
  const Hardware hardware = test_hardware(2.5e9);
  NativeCoarseDieDomain domain("hbm:0", hardware);
  domain.begin_plan_chain("fractional",
                          {hbm_plan(hardware, 81, 4, "fractional")},
                          test_owner(81), 0);
  domain.advance_to(1);
  const json before = domain.progress("fractional");
  const double progress = before.at("active_execution").at("activities")
                              .at(0).at("progress").get<double>();
  require(std::abs(progress - 0.625) < 1.0e-12,
          "live snapshot fixture has fractional native progress");

  const json snapshot = json::parse(domain.export_snapshot().dump());
  require(snapshot.at("schema") == "native_coarse_die_domain_snapshot_v1",
          "domain export uses the versioned wrapper schema");
  auto restored = import_domain(snapshot, hardware);
  require(restored->progress("fractional") == before,
          "wire import preserves current fractional Runtime progress");
  require(restored->export_snapshot() == snapshot,
          "wire export after import preserves the full live wrapper snapshot");
  restored->advance_to(2);
  require(domain.progress("fractional").at("runtime_now_ns") == 1,
          "restored Runtime clock advances independently");
  require(restored->progress("fractional").at("runtime_now_ns") == 2,
          "wire import returns an independently mutable Runtime");
}

void wire_snapshot_preserves_chain_boundary_and_completion_consumption() {
  const Hardware hardware = test_hardware();
  NativeCoarseDieDomain domain("hbm:0", hardware);
  domain.begin_plan_chain("boundary",
                          {hbm_plan(hardware, 82, 3, "boundary-first"),
                           hbm_plan(hardware, 83, 2, "boundary-second")},
                          test_owner(82), 0);
  domain.advance_to(3);
  const json boundary = domain.export_snapshot();
  auto restored_boundary = import_domain(boundary, hardware);
  const json boundary_progress = restored_boundary->progress("boundary");
  require(boundary_progress.at("plan_index") == 1 &&
              boundary_progress.at("completed_plan_progress").size() == 1 &&
              boundary_progress.at("active_execution").at("issue_ns") == 3,
          "snapshot at a chain boundary retains the next Plan's exact issue tick");

  restored_boundary->advance_to(5);
  const json pending = restored_boundary->export_snapshot();
  require(pending.at("completion_queue").size() == 1,
          "terminal chain snapshot retains its unconsumed completion");
  auto restored_pending = import_domain(pending, hardware);
  const auto completion = restored_pending->take_completions();
  require(completion.size() == 1 && completion.front().leg_id == "boundary" &&
              completion.front().finish_ns == 5 &&
              !completion.front().cancelled,
          "unconsumed chain completion survives wire import exactly once");

  const json consumed = restored_pending->export_snapshot();
  require(consumed.at("completion_queue").empty(),
          "consumed terminal is represented by an empty queue");
  auto restored_consumed = import_domain(consumed, hardware);
  require(restored_consumed->take_completions().empty(),
          "import does not recreate an already consumed completion");
  require(restored_consumed->progress("boundary").at("finish_ns") == 5,
          "consumed completion does not discard retained terminal progress");
  require(restored_consumed->retire("boundary"),
          "imported consumed terminal can be retired");
}

void wire_snapshot_preserves_live_endpoint_transaction_refcounts() {
  const Hardware hardware = test_hardware();
  NativeCoarseDieDomain domain("hbm:0", hardware);
  const Owner owner = test_owner(84);
  domain.begin_endpoint("endpoint-a", endpoint_spec(840, 840), owner, 0);
  domain.begin_endpoint("endpoint-b", endpoint_spec(841, 840), owner, 0);
  const json snapshot = domain.export_snapshot();
  auto restored = import_domain(snapshot, hardware);
  const json bindings = restored->export_snapshot().at("endpoint_bindings");
  require(bindings.at("endpoint_ids").size() == 2 &&
              bindings.at("transactions").size() == 1 &&
              bindings.at("transactions").at(0).at("live_endpoints") == 2,
          "same-owner endpoints restore their shared transaction refcount");
  require(restored->progress("endpoint-a").at("active_execution").is_object() &&
              restored->progress("endpoint-b").at("active_execution").is_object(),
          "both imported endpoints remain attached to live Runtime handles");
}

void wire_snapshot_import_rejects_unmatched_handles_and_mismatched_plans() {
  const Hardware hardware = test_hardware();
  NativeCoarseDieDomain domain("hbm:0", hardware);
  domain.begin_plan_chain("bound", {hbm_plan(hardware, 85, 5, "bound")},
                          test_owner(85), 0);
  const json original = domain.export_snapshot();

  json orphaned_wrapper = original;
  orphaned_wrapper["runtime"]["executions"].clear();
  require_bad_domain_snapshot(orphaned_wrapper, hardware,
                              "wrapper handle without Runtime execution is rejected");

  json mismatched_plan = original;
  mismatched_plan["entries"][0]["plans"][0] = coarse_die::plan_to_json(
      *hbm_plan(hardware, 86, 7, "different-valid-plan"));
  require_bad_domain_snapshot(mismatched_plan, hardware,
                              "valid but cursor-mismatched Plan is rejected");

  json mismatched_owner = original;
  mismatched_owner["entries"][0]["owner"] = owner_wire(test_owner(999));
  require_bad_domain_snapshot(mismatched_owner, hardware,
                              "valid but Runtime-mismatched owner is rejected");

  require(domain.export_snapshot() == original,
          "failed imports leave the source domain unchanged");
}

void wire_snapshot_import_rejects_binding_and_completion_tampering() {
  const Hardware hardware = test_hardware();
  NativeCoarseDieDomain endpoints("hbm:0", hardware);
  const Owner owner = test_owner(87);
  endpoints.begin_endpoint("endpoint-a", endpoint_spec(870, 870), owner, 0);
  endpoints.begin_endpoint("endpoint-b", endpoint_spec(871, 870), owner, 0);
  json wrong_refcount = endpoints.export_snapshot();
  wrong_refcount["endpoint_bindings"]["transactions"][0]["live_endpoints"] = 1;
  require_bad_domain_snapshot(wrong_refcount, hardware,
                              "endpoint transaction refcount mismatch is rejected");
  json wrong_tx_owner = endpoints.export_snapshot();
  const json impostor_owner = owner_wire(test_owner(999));
  wrong_tx_owner["endpoint_bindings"]["transactions"][0]["owner"] =
      impostor_owner;
  wrong_tx_owner["runtime"]["transaction_owners"][0]["owner"] =
      impostor_owner;
  require_bad_domain_snapshot(wrong_tx_owner, hardware,
                              "endpoint transaction owner mismatch is rejected");

  NativeCoarseDieDomain finished("hbm:0", hardware);
  finished.begin_plan_chain("done", {hbm_plan(hardware, 88, 2, "done")},
                            test_owner(88), 0);
  finished.advance_to(2);
  json duplicate_completion = finished.export_snapshot();
  duplicate_completion["completion_queue"].push_back(
      duplicate_completion["completion_queue"][0]);
  require_bad_domain_snapshot(duplicate_completion, hardware,
                              "duplicate terminal completion is rejected");

  const json raw_runtime = finished.export_runtime_snapshot();
  require_bad_domain_snapshot(raw_runtime, hardware,
                              "raw Runtime snapshot cannot masquerade as wrapper state");

  json nonfinite_counter = finished.export_snapshot();
  nonfinite_counter["runtime"]["stats"]["hbm_busy_ns"] = "nan";
  require_bad_domain_snapshot(nonfinite_counter, hardware,
                              "nonfinite exact Runtime counter is rejected");
}

void wire_snapshot_preserves_cancelled_chain_and_pending_completion() {
  const Hardware hardware = test_hardware();
  NativeCoarseDieDomain domain("hbm:0", hardware);
  domain.begin_plan_chain("cancel-after-first",
                          {hbm_plan(hardware, 89, 2, "cancel-first"),
                           hbm_plan(hardware, 90, 10, "cancel-second")},
                          test_owner(89), 0);
  domain.advance_to(3);
  require(domain.cancel("cancel-after-first", 3),
          "second chain Plan can be cancelled after prior service");
  const json snapshot = domain.export_snapshot();
  auto restored = import_domain(snapshot, hardware);
  const json progress = restored->progress("cancel-after-first");
  require(progress.at("terminal").get<bool>() &&
              progress.at("cancelled").get<bool>() &&
              progress.at("plan_index") == 1 &&
              progress.at("completed_plan_progress").size() == 1 &&
              progress.at("last_progress").at("cancelled").get<bool>(),
          "cancelled chain restores successful prefix plus cancelled last progress");
  const auto completions = restored->take_completions();
  require(completions.size() == 1 && completions.front().cancelled &&
              completions.front().finish_ns == 3,
          "cancelled wrapper completion survives import exactly once");
}

void local_sample_import_restores_active_plan_and_endpoint_bindings() {
  const Hardware hardware = canonical_test_hardware();
  NativeCoarseDieDomain source("hbm:0", hardware);
  const Owner plan_owner = test_owner(201);
  const Owner endpoint_owner = test_owner(202);
  source.begin_plan_chain(
      "source-plan",
      {hbm_plan(hardware, 201, 1000000000000000ULL, "sample-plan")},
      plan_owner, 0);
  source.begin_endpoint(
      "source-endpoint", endpoint_spec(2020, 2020, 1000000000000000ULL),
      endpoint_owner, 0);
  source.advance_to(1);

  const json runtime_snapshot = source.export_runtime_snapshot();
  const json sample = local_sample(hardware, runtime_snapshot, 0);
  std::map<std::uint64_t, std::string> handle_to_leg_id;
  for (const auto& execution : runtime_snapshot.at("executions")) {
    const std::uint64_t handle = execution.at("handle").get<std::uint64_t>();
    const bool endpoint =
        execution.at("plan").at("task").value("schema", std::string()) ==
        "coarse_die_endpoint_v2";
    handle_to_leg_id.emplace(
        handle, endpoint ? "external:opaque-endpoint-leg"
                         : "external:opaque-plan-leg");
  }
  require(handle_to_leg_id.size() == 2,
          "local sample fixture has one active Plan and one endpoint");
  const json sample_before = sample;
  auto imported = NativeCoarseDieDomain::import_local_sample(
      "hbm:0", std::make_shared<const Hardware>(hardware), 0, sample,
      handle_to_leg_id);
  require(sample == sample_before,
          "successful local-sample import leaves the source sample unchanged");
  require(imported->export_runtime_snapshot() == runtime_snapshot,
          "active handles reuse the canonical sampled Runtime without recompilation");
  const json plan_progress =
      imported->progress("external:opaque-plan-leg");
  const json endpoint_progress =
      imported->progress("external:opaque-endpoint-leg");
  const std::uint64_t plan_handle = [&]() {
    for (const auto& [handle, leg_id] : handle_to_leg_id) {
      if (leg_id == "external:opaque-plan-leg") return handle;
    }
    throw std::runtime_error("explicit Plan leg mapping was lost");
  }();
  const std::uint64_t endpoint_handle = [&]() {
    for (const auto& [handle, leg_id] : handle_to_leg_id) {
      if (leg_id == "external:opaque-endpoint-leg") return handle;
    }
    throw std::runtime_error("explicit endpoint leg mapping was lost");
  }();
  require(plan_progress.at("owner") == owner_wire(plan_owner) &&
              plan_progress.at("active_execution").at("handle") ==
                  plan_handle &&
              plan_progress.at("state") == "running",
          "explicit external Plan leg preserves its sampled Owner and handle");
  require(endpoint_progress.at("owner") == owner_wire(endpoint_owner) &&
              endpoint_progress.at("active_execution").at("handle") ==
                  endpoint_handle &&
              endpoint_progress.at("state") == "running",
          "explicit external endpoint leg preserves its sampled Owner and handle");
  const json endpoint_bindings =
      imported->export_snapshot().at("endpoint_bindings");
  require(endpoint_bindings.at("endpoint_ids").size() == 1 &&
              endpoint_bindings.at("endpoint_ids").at(0).at("leg_id") ==
                  "external:opaque-endpoint-leg" &&
              endpoint_bindings.at("transactions").at(0).at("live_endpoints") ==
                  1,
          "active endpoint registry and transaction reference are restored");

  source.advance_to(7);
  imported->advance_to(7);
  json imported_plan_progress = imported->progress("external:opaque-plan-leg");
  json source_plan_progress = source.progress("source-plan");
  imported_plan_progress.erase("leg_id");
  source_plan_progress.erase("leg_id");
  json imported_endpoint_progress =
      imported->progress("external:opaque-endpoint-leg");
  json source_endpoint_progress = source.progress("source-endpoint");
  imported_endpoint_progress.erase("leg_id");
  source_endpoint_progress.erase("leg_id");
  require(imported->export_runtime_snapshot() ==
              source.export_runtime_snapshot() &&
              imported_plan_progress == source_plan_progress &&
              imported_endpoint_progress == source_endpoint_progress,
          "imported active Plan and endpoint continue identically on copied Runtime state");
}

void local_sample_terminal_handle_becomes_one_pending_wrapper_completion() {
  const Hardware hardware = canonical_test_hardware();
  coarse_die::Runtime source(hardware);
  const Owner owner = test_owner(203);
  const std::uint64_t handle =
      source.submit_endpoint(endpoint_spec(2030, 2030, 1), owner, 0);
  const auto terminal_tick = source.next_event();
  require(terminal_tick.has_value(),
          "terminal sample fixture exposes the endpoint completion tick");
  source.advance_to(*terminal_tick);
  const json raw_progress = source.query_progress(handle);
  require(raw_progress.at("terminal").get<bool>() &&
              raw_progress.at("finish_ns") == *terminal_tick,
          "source Runtime is sampled at the same tick as its terminal callback");
  const json sample = local_sample(hardware, source.export_snapshot(), 0);
  const std::map<std::uint64_t, std::string> mapping = {
      {handle, "external:terminal-endpoint"}};
  auto imported = NativeCoarseDieDomain::import_local_sample(
      "hbm:0", std::make_shared<const Hardware>(hardware), 0, sample,
      mapping);
  const json snapshot = imported->export_snapshot();
  require(snapshot.at("runtime").at("executions").empty() &&
              snapshot.at("runtime").at("endpoint_registry").empty() &&
              snapshot.at("runtime").at("transaction_owners").empty(),
          "terminal handle is removed from the imported private Runtime copy");
  require(snapshot.at("entries").size() == 1 &&
              snapshot.at("entries").at(0).at("terminal").get<bool>() &&
              snapshot.at("entries").at(0).at("handle").is_null() &&
              snapshot.at("entries").at(0).at("last_progress").at("handle") ==
                  handle &&
              snapshot.at("entries").at(0).at("last_progress") == raw_progress,
          "terminal endpoint is retained as exact wrapper progress with its original handle");
  require(snapshot.at("completion_queue").size() == 1 &&
              snapshot.at("completion_queue").at(0).at("leg_id") ==
                  "external:terminal-endpoint" &&
              snapshot.at("completion_queue").at(0).at("finish_ns") ==
                  *terminal_tick,
          "undispatched same-tick terminal becomes exactly one pending completion");
  require(imported->take_completions().size() == 1 &&
              imported->take_completions().empty(),
          "sampled terminal completion is consumable exactly once");

  coarse_die::Runtime tied_source(hardware);
  const Owner tied_owner = test_owner(204);
  const std::uint64_t lower_handle = tied_source.submit_endpoint(
      endpoint_spec(2040, 2040, 1), tied_owner, 0);
  const std::uint64_t higher_handle = tied_source.submit_endpoint(
      endpoint_spec(2041, 2041, 1), tied_owner, 0);
  const auto tied_tick = tied_source.next_event();
  require(tied_tick.has_value(), "tied terminal fixture exposes a finish tick");
  tied_source.advance_to(*tied_tick);
  require(tied_source.query_progress(lower_handle).at("finish_ns") ==
              tied_source.query_progress(higher_handle).at("finish_ns"),
          "tied terminal fixture completes both handles at the same tick");
  const json tied_sample =
      local_sample(hardware, tied_source.export_snapshot(), 0);
  const std::map<std::uint64_t, std::string> reversed_leg_names = {
      {lower_handle, "z-external-leg"},
      {higher_handle, "a-external-leg"}};
  auto tied_import = NativeCoarseDieDomain::import_local_sample(
      "hbm:0", std::make_shared<const Hardware>(hardware), 0, tied_sample,
      reversed_leg_names);
  const json tied_queue =
      tied_import->export_snapshot().at("completion_queue");
  require(tied_queue.size() == 2 &&
              tied_queue.at(0).at("leg_id") == "z-external-leg" &&
              tied_queue.at(1).at("leg_id") == "a-external-leg",
          "same-tick imported callbacks retain ascending Runtime handle order, not leg-name order");
}

void local_sample_import_rejects_binding_and_identity_tampering() {
  const Hardware hardware = canonical_test_hardware();
  NativeCoarseDieDomain source("hbm:0", hardware);
  source.begin_plan_chain("tamper-plan",
                          {hbm_plan(hardware, 211, 1000000000000000ULL,
                                    "tamper")},
                          test_owner(211), 0);
  source.begin_plan_chain("tamper-plan-2",
                          {hbm_plan(hardware, 212, 1000000000000000ULL,
                                    "tamper-2")},
                          test_owner(212), 0);
  source.advance_to(1);
  const json runtime_snapshot = source.export_runtime_snapshot();
  const json sample = local_sample(hardware, runtime_snapshot, 0);
  const std::uint64_t first_handle = runtime_snapshot.at("executions")
                                         .at(0).at("handle")
                                         .get<std::uint64_t>();
  const std::uint64_t second_handle = runtime_snapshot.at("executions")
                                          .at(1).at("handle")
                                          .get<std::uint64_t>();
  const std::map<std::uint64_t, std::string> valid_mapping = {
      {first_handle, "external:tamper-plan-a"},
      {second_handle, "external:tamper-plan-b"}};

  require_bad_local_sample(sample, hardware, {}, 0,
                           "missing explicit handle-to-leg mapping is rejected");
  require_bad_local_sample(sample, hardware,
                           {{first_handle, "external:tamper-plan-a"},
                            {99, "external:extra-leg"}},
                           0, "extra explicit handle mapping is rejected");
  require_bad_local_sample(sample, hardware,
                           {{first_handle, "same-leg"},
                            {second_handle, "same-leg"}},
                           0, "duplicate explicit leg id is rejected");
  require_bad_local_sample(sample, hardware,
                           {{first_handle, ""},
                            {second_handle, "external:tamper-plan-b"}},
                           0, "empty explicit leg id is rejected");
  require_bad_local_sample(sample, hardware, valid_mapping, 1,
                           "sample rank mismatch is rejected");

  json wrong_hardware = sample;
  wrong_hardware["hardware_fingerprint"] = "sha256:wrong-hardware";
  require_bad_local_sample(wrong_hardware, hardware, valid_mapping, 0,
                           "tampered sample Hardware identity is rejected");
  json wrong_hardware_config = sample;
  wrong_hardware_config["hardware_config"]["performance_profile"]
                       ["eta_hbm_matrix"] = 0.71;
  require_bad_local_sample(wrong_hardware_config, hardware, valid_mapping, 0,
                           "valid but different sampled hardware config is rejected");

  json wrong_owner = sample;
  wrong_owner["completion_bindings"][0]["owner"] = owner_wire(test_owner(999));
  require_bad_local_sample(wrong_owner, hardware, valid_mapping, 0,
                           "completion binding Owner mismatch is rejected");

  json wrong_tick = sample;
  wrong_tick["sampled_at_ns"] =
      wrong_tick.at("sampled_at_ns").get<std::uint64_t>() + 1;
  require_bad_local_sample(wrong_tick, hardware, valid_mapping, 0,
                           "sample and Runtime tick mismatch is rejected");

  json wrong_state = sample;
  wrong_state["completion_bindings"][0]["state_at_sample"] = "finished";
  require_bad_local_sample(wrong_state, hardware, valid_mapping, 0,
                           "completion binding state mismatch is rejected");

  json missing_completion = sample;
  missing_completion["completion_bindings"].clear();
  require_bad_local_sample(missing_completion, hardware, valid_mapping, 0,
                           "missing completion binding is rejected");

  json duplicate_completion = sample;
  duplicate_completion["completion_bindings"].push_back(
      duplicate_completion["completion_bindings"][0]);
  require_bad_local_sample(duplicate_completion, hardware, valid_mapping, 0,
                           "duplicate completion binding is rejected");

  json dispatched_sample = sample;
  dispatched_sample["callbacks_dispatched"] = true;
  require_bad_local_sample(dispatched_sample, hardware, valid_mapping, 0,
                           "sample after callback dispatch is rejected");

  json terminal_sample = local_sample(hardware, [&]() {
    coarse_die::Runtime terminal_runtime(hardware);
    (void)terminal_runtime.submit_endpoint(endpoint_spec(2120, 2120, 1),
                                           test_owner(212), 0);
    const auto finish = terminal_runtime.next_event();
    require(finish.has_value(), "terminal tamper fixture has a finish tick");
    terminal_runtime.advance_to(*finish);
    return terminal_runtime.export_snapshot();
  }(), 0);
  json wrong_terminal = terminal_sample;
  wrong_terminal["completion_bindings"][0]["finish_ns_at_sample"] = 999;
  const auto terminal_handle = terminal_sample.at("completion_bindings")
                                   .at(0).at("handle").get<std::uint64_t>();
  require_bad_local_sample(wrong_terminal, hardware,
                           {{terminal_handle, "external:terminal"}}, 0,
                           "terminal completion timestamp mismatch is rejected");
}

}  // namespace

// A complete first Plan must not hide an unavailable later Plan. Importing a
// diagnostic Runtime/sample must also respect the production consumer boundary.
void unavailable_prediction_rejects_fresh_and_restored_execution() {
  const auto hardware=canonical_test_hardware();
  const auto expect_unavailable=[](auto&& operation,const char* message) {
    bool rejected=false;
    try {operation();}catch(const std::exception& e) {
      rejected=std::string(e.what()).find("prediction_not_available")!=std::string::npos;
    }
    require(rejected,message);
  };
  for(const char* field:{"prediction_status","alone_estimate_status"}) {
    const auto first=hbm_plan(hardware,110,1000000,"available");
    auto body=std::make_shared<Plan>(*hbm_plan(hardware,111,1000000,"body"));
    body->summary[field]="not_available";body->fingerprint.clear();
    body->fingerprint=coarse_die::plan_to_json(*body).at("fingerprint").get<std::string>();
    NativeCoarseDieDomain domain("hbm:0",hardware);
    domain.begin_plan_chain("blocker",{first},test_owner(110),0);
    const auto before=domain.export_snapshot();
    expect_unavailable([&]{domain.begin_plan_chain("incomplete",{first,body},test_owner(111),5);},
        "an unavailable later Plan must reject the whole chain before live advance");
    require(domain.export_snapshot()==before,"failed prediction leaves clock, work and bindings unchanged");

    NativeCoarseDieDomain source("hbm:0",hardware);
    source.begin_plan_chain("restore",{first,first},test_owner(112),0);
    auto snapshot=source.export_snapshot();
    snapshot["entries"][0]["plans"][1]=coarse_die::plan_to_json(*body);
    const auto wire_before=snapshot;
    expect_unavailable([&]{(void)import_domain(snapshot,hardware);},
        "snapshot restore must check future Plans as well as the active Plan");
    require(snapshot==wire_before,"failed prediction import leaves the source snapshot intact");

    coarse_die::Runtime diagnostics(hardware);
    const auto handle=diagnostics.submit(body,test_owner(113),0);
    const auto sample=local_sample(hardware,diagnostics.export_snapshot(),0);
    expect_unavailable([&]{(void)NativeCoarseDieDomain::import_local_sample(
        "hbm:0",std::make_shared<const Hardware>(hardware),0,sample,{{handle,"body"}});},
        "body-only public diagnostics must not become a production local-sample completion");
  }
}

int main() {
  try {
    unavailable_prediction_rejects_fresh_and_restored_execution();
    chain_successor_starts_at_prior_terminal_tick();
    cancellation_keeps_served_work_and_emits_one_terminal();
    invalid_later_plan_is_rejected_before_domain_advance();
    invalid_endpoint_is_rejected_before_domain_advance();
    endpoint_completion_releases_native_endpoint_binding();
    same_tick_chain_dispatch_is_stable_across_registration_order();
    capacity_updates_reject_invalid_events_before_live_advance();
    capacity_updates_retime_shared_comp_and_endpoint_and_survive_forks();
    capacity_event_at_chain_terminal_uses_old_rate_then_updates_successor();
    exact_json_nanosecond_boundary_is_enforced();
    fork_copies_runtime_and_chain_bindings_without_aliasing();
    wire_snapshot_roundtrips_fractional_live_progress();
    wire_snapshot_preserves_chain_boundary_and_completion_consumption();
    wire_snapshot_preserves_live_endpoint_transaction_refcounts();
    wire_snapshot_import_rejects_unmatched_handles_and_mismatched_plans();
    wire_snapshot_import_rejects_binding_and_completion_tampering();
    wire_snapshot_preserves_cancelled_chain_and_pending_completion();
    local_sample_import_restores_active_plan_and_endpoint_bindings();
    local_sample_terminal_handle_becomes_one_pending_wrapper_completion();
    local_sample_import_rejects_binding_and_identity_tampering();
    std::cout << "PASS native_coarse_die_domain_test\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL native_coarse_die_domain_test: " << error.what() << '\n';
    return 1;
  }
}
