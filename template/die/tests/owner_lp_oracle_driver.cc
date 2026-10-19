// Emits public Runtime observations for validation/owner_lp_oracle.py.  The
// independent Python Fraction solver consumes only this JSON interface.
#include "test_support.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {
constexpr std::uint64_t kNs = 1000000000ULL;
using json = nlohmann::json;

std::array<double, resource_count> capacities(double hbm0, double compute = 0) {
  std::array<double, resource_count> result{};
  result[HBM0] = hbm0;
  result[COMPUTE_GATE] = compute;
  return result;
}

std::shared_ptr<const Plan> one_epoch_plan(
    const Hardware& hardware, const std::string& name,
    std::vector<Activity> activities,
    std::array<double, 2> credit = {0.5, 0.5},
    std::array<double, 2> credit_capacity = {1.0, 1.0}) {
  Epoch epoch;
  epoch.name = name;
  epoch.credit_initial = credit;
  epoch.credit_capacity = credit_capacity;
  epoch.activities = std::move(activities);
  return manual_plan(hardware, {std::move(epoch)}, name);
}

Activity pipeline_load() {
  auto load = activity(1, Role::Load, HBM0, 2.0);
  load.hbm_bytes[0][0] = 2;
  load.canonical_q = 2.0;
  load.private_time_s = 0.5;
  load.credit_delta = {1.0, 0.0};
  return load;
}

Activity pipeline_compute() {
  auto compute = activity(2, Role::Compute, COMPUTE_GATE, 4.0);
  compute.canonical_q = 5.0;
  compute.private_time_s = 0.5;
  compute.credit_delta = {-1.0, 0.0};
  return compute;
}

std::string role_name(Role role) {
  switch (role) {
    case Role::Load: return "load";
    case Role::Compute: return "compute";
    case Role::Store: return "store";
    case Role::Endpoint: return "endpoint";
  }
  throw std::runtime_error("unknown activity role");
}

json observe(const std::string& label, const Runtime& runtime,
             const std::map<std::uint64_t, std::shared_ptr<const Plan>>& plans) {
  const json snapshot = runtime.export_snapshot();
  const std::uint64_t now = snapshot.at("now_ns").get<std::uint64_t>();
  json sample = {{"label", label}, {"now_ns", now},
                 {"capacities", snapshot.at("capacities")},
                 {"executions", json::array()}};
  for (const std::uint64_t handle : runtime.execution_handles()) {
    const json progress = runtime.query_progress(handle);
    if (progress.at("terminal").get<bool>() ||
        progress.at("state") == "queued") continue;
    const auto plan_found = plans.find(handle);
    require(plan_found != plans.end(), "probe has Plan metadata for every handle");
    const auto& plan = *plan_found->second;
    const std::size_t current_epoch =
        progress.at("epoch_index").get<std::size_t>();
    if (current_epoch >= plan.epochs.size()) continue;
    const auto snap_found = std::find_if(
        snapshot.at("executions").begin(), snapshot.at("executions").end(),
        [handle](const json& item) {
          return item.at("handle").get<std::uint64_t>() == handle;
        });
    require(snap_found != snapshot.at("executions").end(),
            "snapshot retains every queryable execution");
    const json snap_execution = *snap_found;
    json execution = {{"handle", handle}, {"owner", progress.at("owner")},
                      {"credit", snap_execution.at("credit")},
                      {"credit_capacity", snap_execution.at("credit_capacity")},
                      {"activities", json::array()}};
    const json& states = progress.at("activities");
    for (const json& state : states) {
      if (state.at("epoch_index").get<std::size_t>() != current_epoch ||
          state.at("done").get<bool>()) continue;
      const auto& definitions = plan.epochs[current_epoch].activities;
      const auto definition = std::find_if(
          definitions.begin(), definitions.end(), [&](const Activity& item) {
            return item.id == state.at("activity_id").get<std::uint64_t>();
          });
      require(definition != definitions.end(), "progress activity maps to its Plan definition");
      bool service = false;
      json work = json::array();
      for (double amount : definition->work) {
        service = service || amount > 0.0;
        work.push_back(amount);
      }
      if (!service) continue;
      execution["activities"].push_back({
          {"activity_id", definition->id},
          {"role", role_name(definition->role)},
          {"q", definition->canonical_q},
          {"work", std::move(work)},
          {"private_time", definition->private_time_s},
          {"delta", definition->credit_delta},
          {"progress", state.at("progress")},
          {"rate", state.at("rate")},
          {"alpha", state.at("alpha")}});
    }
    if (!execution.at("activities").empty())
      sample["executions"].push_back(std::move(execution));
  }
  return sample;
}

std::unique_ptr<Runtime> check_snapshot(const Hardware& hardware,
                                       Runtime& runtime,
                                       const std::vector<std::uint64_t>& handles) {
  auto copy = Runtime::import_snapshot(hardware, runtime.export_snapshot());
  for (const auto handle : handles)
    require(copy->query_progress(handle) == runtime.query_progress(handle),
            "snapshot restore preserves per-activity progress and allocation");
  return copy;
}

void add_sample(json& output, const std::string& label, const Runtime& runtime,
                const std::map<std::uint64_t, std::shared_ptr<const Plan>>& plans) {
  output.push_back(observe(label, runtime, plans));
}

void static_mixed_and_permutation(json& output) {
  const Hardware hardware = manual_hardware(capacities(3.0, 6.0));
  json forward = json::array();
  json reverse = json::array();
  for (const bool reversed : {false, true}) {
    Runtime runtime(hardware);
    std::map<std::uint64_t, std::shared_ptr<const Plan>> plans;
    std::vector<std::uint64_t> handles(3);
    for (int step = 0; step < 3; ++step) {
      const int index = reversed ? 2 - step : step;
      const std::uint64_t node = static_cast<std::uint64_t>(index + 1);
      std::array<double, 2> credit{
          index == 0 ? 0.0 : index == 1 ? 1.0e-10 : 1.0, 0.5};
      std::vector<Activity> activities;
      if (index < 2) {
        activities = {pipeline_load(), pipeline_compute()};
      } else {
        auto other = activity(3, Role::Load, HBM0, 1.0);
        other.hbm_bytes[0][0] = 1;
        other.canonical_q = 7.0;
        other.private_time_s = 0.5;
        other.credit_delta = {-1.0, 0.0};
        activities = {other};
      }
      const auto plan = one_epoch_plan(
          hardware, "mixed-owner-block-" + std::to_string(index),
          std::move(activities), credit);
      const auto handle = runtime.submit(plan,
          owner(100 + node, node, node), 0);
      handles[static_cast<std::size_t>(index)] = handle;
      plans.emplace(handle, plan);
    }
    add_sample(reversed ? reverse : forward,
               reversed ? "mixed-reverse" : "mixed-forward", runtime, plans);
  }
  output.insert(output.end(), forward.begin(), forward.end());
  output.insert(output.end(), reverse.begin(), reverse.end());
}

void private_bound_not_multiplied(json& output) {
  const Hardware hardware = manual_hardware(capacities(100.0));
  auto only = activity(1, Role::Load, HBM0, 1.0);
  only.hbm_bytes[0][0] = 1;
  only.canonical_q = 7.0;
  only.private_time_s = 0.5;
  const auto plan = one_epoch_plan(hardware, "private-per-owner", {only});
  Runtime runtime(hardware);
  std::map<std::uint64_t, std::shared_ptr<const Plan>> plans;
  for (std::uint64_t id = 1; id <= 2; ++id) {
    const auto handle = runtime.submit(plan, owner(id, id, id), 0);
    plans.emplace(handle, plan);
  }
  add_sample(output, "private-not-multiplied-runtime", runtime, plans);
}

void curved_ratio_boundaries(json& output) {
  const Hardware hardware = manual_hardware(capacities(3.0, 2.0));
  for (const unsigned count : {2U, 3U}) {
    auto load = activity(1, Role::Load, HBM0, 2.0);
    load.canonical_q = 2.0;
    load.private_time_s = 0.25;
    load.credit_delta = {1.0, 0.0};
    auto compute = activity(2, Role::Compute, COMPUTE_GATE, 4.0);
    compute.canonical_q = 5.0;
    compute.private_time_s = 0.5;
    compute.credit_delta = {-1.0, count == 3 ? 1.0 : 0.0};
    std::vector<Activity> activities{load, compute};
    if (count == 3) {
      auto store = activity(3, Role::Store, HBM0, 1.0);
      store.canonical_q = 3.0;
      store.private_time_s = 0.125;
      store.credit_delta = {0.0, -1.0};
      activities.push_back(store);
    }
    for (auto& item : activities)
      item.credit_curve = {{0.0, {0.0, 0.0}}, {1.0, item.credit_delta}};
    const auto plan = one_epoch_plan(hardware, "curved-ratio-" +
        std::to_string(count), std::move(activities), {0.0, 0.0});
    Runtime runtime(hardware);
    const auto handle = runtime.submit(plan, owner(count, count, count), 0);
    require(runtime.last_arbitration().path ==
                ArbitrationDiagnostics::Path::Ratio,
            "curved tangent boundary exercises the ratio allocation path");
    const std::map<std::uint64_t, std::shared_ptr<const Plan>> plans{{handle, plan}};
    add_sample(output, "curved-ratio-" + std::to_string(count), runtime, plans);
    auto restored = check_snapshot(hardware, runtime, {handle});
    add_sample(output, "curved-ratio-restored-" + std::to_string(count),
               *restored, plans);
    runtime.update_capacity(HBM0, 0.5, 0);
    add_sample(output, "curved-ratio-capacity-" + std::to_string(count),
               runtime, plans);
  }
}

void split_fair_owner_request_metadata(json& output) {
  const Hardware hardware = manual_hardware(capacities(6.0));
  Runtime runtime(hardware);
  std::map<std::uint64_t, std::shared_ptr<const Plan>> plans;
  auto first = activity(1, Role::Load, HBM0, 2.0);
  first.hbm_bytes[0][0] = 2;
  first.canonical_q = 2.0;
  auto second = first;
  second.id = 2;
  second.canonical_q = 5.0;
  auto other = first;
  other.id = 3;
  other.canonical_q = 7.0;
  const auto plan_a = one_epoch_plan(hardware, "owner-split-first", {first});
  const auto plan_b = one_epoch_plan(hardware, "owner-split-second", {second});
  const auto plan_c = one_epoch_plan(hardware, "owner-split-peer", {other});
  const auto ha = runtime.submit(plan_a, owner(100, 1, 10), 0);
  const auto hb = runtime.submit(plan_b, owner(999, 1, 10), 0);
  const auto hc = runtime.submit(plan_c, owner(1000, 2, 20), 0);
  plans.emplace(ha, plan_a);
  plans.emplace(hb, plan_b);
  plans.emplace(hc, plan_c);
  add_sample(output, "split-fair-owner-different-request-ids", runtime, plans);
}

void arrival_completion_and_restore(json& output) {
  const Hardware hardware = manual_hardware(capacities(1.0, 2.0));
  const std::vector<Activity> block{pipeline_load(), pipeline_compute()};
  const auto plan_a = one_epoch_plan(hardware, "arrival-early", block,
                                     {0.0, 0.5});
  const auto plan_b = one_epoch_plan(hardware, "arrival-late", block,
                                     {0.0, 0.5});
  Runtime runtime(hardware);
  std::map<std::uint64_t, std::shared_ptr<const Plan>> plans;
  const auto early = runtime.submit(plan_a, owner(1, 1, 1), 0);
  plans.emplace(early, plan_a);
  add_sample(output, "arrival-before-join", runtime, plans);
  runtime.advance_to(kNs);
  require_close(runtime.query_progress(early).at("activities").at(0)
                    .at("progress").get<double>(), 0.5, 1e-12,
                "early owner integrates one second before the late arrival");
  const auto late = runtime.submit(plan_b, owner(2, 2, 2), kNs);
  plans.emplace(late, plan_b);
  const std::vector<std::uint64_t> handles{early, late};
  add_sample(output, "arrival-join-phase-offset", runtime, plans);
  auto fork = check_snapshot(hardware, runtime, handles);
  add_sample(output, "arrival-join-snapshot-restore", *fork, plans);

  runtime.advance_to(3 * kNs);
  require(runtime.terminal_finish_ns(early) == 3 * kNs,
          "early mirror exits at its own completion event");
  const auto late_progress = runtime.query_progress(late).at("activities");
  for (const auto& activity_state : late_progress) {
    require_close(activity_state.at("progress").get<double>(), 0.5,
                  1e-12, "late mirror retains integrated stage service at member removal");
    require_close(activity_state.at("rate").get<double>(), 0.5,
                  1e-12, "surviving owner receives its single-owner stage rate");
  }
  add_sample(output, "arrival-member-completed", runtime, plans);
  runtime.advance_to(4 * kNs);
  require(runtime.drained() && runtime.terminal_finish_ns(late) == 4 * kNs,
          "late mirror completes one second after the first mirror");
  const auto counters = runtime.counters();
  require(counters.at("hbm_busy_ns_exact") == "4000000000" &&
              counters.at("hbm_shared_ns_exact") == "2000000000",
          "arrival and completion preserve four-second busy and two-second shared unions");
}

void capacity_cancel_and_restore(json& output) {
  const Hardware hardware = manual_hardware(capacities(2.0));
  auto bytes = activity(1, Role::Load, HBM0, 2.0);
  bytes.hbm_bytes[0][0] = 2;
  bytes.canonical_q = 2.0;
  const auto plan = one_epoch_plan(hardware, "capacity-and-cancel", {bytes});
  Runtime runtime(hardware);
  std::map<std::uint64_t, std::shared_ptr<const Plan>> plans;
  const auto a = runtime.submit(plan, owner(1, 1, 1), 0);
  const auto b = runtime.submit(plan, owner(2, 2, 2), 0);
  plans.emplace(a, plan);
  plans.emplace(b, plan);
  add_sample(output, "capacity-before-change", runtime, plans);

  runtime.update_capacity(HBM0, 1.0, kNs);
  add_sample(output, "capacity-after-change", runtime, plans);
  const std::vector<std::uint64_t> handles{a, b};
  auto fork = check_snapshot(hardware, runtime, handles);
  add_sample(output, "capacity-snapshot-restore", *fork, plans);

  runtime.advance_to(2 * kNs);
  runtime.cancel(b, 2 * kNs);
  require(runtime.query_progress(b).at("cancelled").get<bool>(),
          "second owner is cancelled at the requested boundary");
  add_sample(output, "cancel-reallocation", runtime, plans);
  auto after_cancel = check_snapshot(hardware, runtime, handles);
  add_sample(output, "cancel-snapshot-restore", *after_cancel, plans);

  runtime.advance_to(2500000000ULL);
  require(runtime.terminal_finish_ns(a) == 2500000000ULL,
          "remaining quarter progress completes after half a second");
  const auto counters = runtime.counters();
  require(counters.at("hbm_busy_ns_exact") == "2500000000" &&
              counters.at("hbm_shared_ns_exact") == "2000000000",
          "capacity change and cancellation preserve busy-union accounting");
}

json run() {
  json output = json::array();
  static_mixed_and_permutation(output);
  private_bound_not_multiplied(output);
  curved_ratio_boundaries(output);
  split_fair_owner_request_metadata(output);
  arrival_completion_and_restore(output);
  capacity_cancel_and_restore(output);
  return output;
}
}  // namespace

int main() {
  try {
    std::cout << run().dump() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL owner_lp_oracle_driver: " << error.what() << '\n';
    return 1;
  }
}
