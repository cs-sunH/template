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
    wire_snapshot_roundtrips_fractional_live_progress();
    wire_snapshot_preserves_chain_boundary_and_completion_consumption();
    wire_snapshot_preserves_live_endpoint_transaction_refcounts();
    wire_snapshot_import_rejects_unmatched_handles_and_mismatched_plans();
    wire_snapshot_import_rejects_binding_and_completion_tampering();
    wire_snapshot_preserves_cancelled_chain_and_pending_completion();
    std::cout << "PASS native_coarse_die_domain_test\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL native_coarse_die_domain_test: " << error.what() << '\n';
    return 1;
  }
}
