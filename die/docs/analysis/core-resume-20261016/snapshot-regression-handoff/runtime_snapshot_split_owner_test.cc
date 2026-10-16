#include "test_support.hh"
#include "coarse_die/Capi.h"
#include <functional>
#include <memory>
#include <vector>

using namespace coarse_die;
using namespace coarse_die_test;
namespace {
void ok(int status) { require(status == DIE_OK, die_coarse_last_error_v2()); }
json take(const char* value) {
  require(value != nullptr, die_coarse_last_error_v2());
  std::unique_ptr<const char, decltype(&die_coarse_free_result_v2)>
      owned(value, die_coarse_free_result_v2);
  return json::parse(value);
}
json snapshot(DieHandle runtime) {
  return take(die_coarse_export_snapshot_v2(runtime));
}
json next_event(DieHandle runtime) {
  return take(die_coarse_next_event_v2(runtime)).at("next_event_ns");
}
DieHandle imported(DieHandle model, DieHandle parent, const json& saved) {
  const auto before = snapshot(parent);
  DieHandle result = 123456;
  const int status = die_coarse_import_snapshot_v2(model, saved.dump().c_str(), &result);
  const std::string error = die_coarse_last_error_v2();
  require(snapshot(parent) == before, "import leaves the complete parent unchanged");
  require(status == DIE_OK || result == 0, "failed import clears its output handle");
  require(status == DIE_OK, "split-owner import: " + error);
  require(snapshot(result) == saved, "import preserves every saved field and published rate");
  return result;
}
DieHandle forked(DieHandle parent) {
  const auto before = snapshot(parent);
  DieHandle result = 123456;
  const int status = die_coarse_fork_runtime_v2(parent, &result);
  const std::string error = die_coarse_last_error_v2();
  require(snapshot(parent) == before, "fork leaves the complete parent unchanged");
  require(status == DIE_OK || result == 0, "failed fork clears its output handle");
  require(status == DIE_OK, "split-owner fork: " + error);
  require(snapshot(result) == before, "fork retains the source allocation exactly");
  return result;
}
void reject(DieHandle model, DieHandle parent, const json& bad, const char* label, int expected_status = DIE_INVALID_INPUT) {
  const auto before = snapshot(parent);
  DieHandle result = 123456;
  const int status = die_coarse_import_snapshot_v2(model, bad.dump().c_str(), &result);
  require(status == expected_status && result == 0, std::string(label) + " must reject atomically");
  require(snapshot(parent) == before, std::string(label) + " leaves parent unchanged");
}
void compare_and_drain(const std::vector<DieHandle>& worlds) {
  for (unsigned steps = 0; steps < 64; ++steps) {
    const auto saved = snapshot(worlds.front());
    const auto next = next_event(worlds.front());
    for (auto world : worlds) {
      require(snapshot(world) == saved, "source/import/fork full snapshots agree at each event");
      require(next_event(world) == next, "source/import/fork next events agree");
    }
    if (next.is_null()) {
      for (const auto& execution : saved.at("executions"))
        require(execution.at("terminal").get<bool>(), "missing event must not hide a stalled live execution");
      return;
    }
    for (auto world : worlds) ok(die_coarse_advance_to_v2(world, next.get<std::uint64_t>()));
  }
  throw std::runtime_error("fixed split-owner fixture exceeded its event bound");
}
// A legitimate neighboring inward rate still obeys the original one-ULP rule.
// An outward neighbor at a finite-credit boundary must remain rejected.
void check_outward_tangent() {
  std::array<double, resource_count> rates{};
  rates[COMPUTE_GATE] = rates[HBM0] = 1;
  const auto hw = manual_hardware(rates);
  Epoch epoch; epoch.name = "one_ulp_outward_tangent"; epoch.credit_capacity = {{0, 1}};
  auto compute = activity(1, Role::Compute, COMPUTE_GATE, 1000000);
  compute.credit_delta = {{0, 2000000}};
  compute.credit_curve = {{0, {{0, 0}}}, {.5, {{0, 0}}}, {1, compute.credit_delta}};
  auto store = activity(2, Role::Store, HBM0, 2000000);
  store.hbm_bytes[0][1] = 2000000;
  store.canonical_q = 2000000;
  store.credit_delta = {{0, -2000000}};
  store.credit_curve = {{0, {{0, 0}}}, {1, store.credit_delta}};
  epoch.activities = {compute, store};
  Runtime source(hw);
  source.submit(manual_plan(hw, {epoch}, "outward_neighbor"), owner(1, 1, 1), 0);
  for (unsigned event = 0; event < 20; ++event) {
    const auto saved = source.export_snapshot();
    if (saved.at("executions").at(0).at("credit").at(1).get<double>() == 1) {
      auto bad = saved;
      auto& rate = bad["executions"][0]["activities"][0]["rate"];
      rate = std::nextafter(rate.get<double>(), std::numeric_limits<double>::infinity());
      bool rejected = false;
      try { (void)Runtime::import_snapshot(hw, bad); }
      catch (const std::invalid_argument&) { rejected = true; }
      require(rejected, "one-ULP outward finite-credit tangent remains invalid");
      require(source.export_snapshot() == saved, "outward import failure leaves source unchanged");
      return;
    }
    const auto next = source.next_event();
    require(next.has_value(), "outward fixture reaches a finite-credit boundary");
    source.advance_to(*next);
  }
  throw std::runtime_error("outward fixture did not reach its finite-credit boundary");
}
void verify() {
#ifdef COARSE_DIE_SPLIT_FIXTURE
  const auto data = read_json(COARSE_DIE_SPLIT_FIXTURE);
#else
  const auto data = read_json(fixture("snapshot_split_owner.json"));
#endif
  require(data.at("schema") == "coarse_die_snapshot_split_owner_fixture_v1", "fixture schema");
  DieHandle model = 0;
  ok(die_coarse_create_model_v2(data.at("hardware").dump().c_str(), &model));
  const auto compiled = take(die_coarse_compile_batch_v2(model,
      json{{"specs", data.at("tasks")}}.dump().c_str()));
  require(compiled.at("results").size() == 2, "two actual GEMM/Norm Plans compile");
  const auto make_world = [&]() {
    DieHandle runtime = 0;
    ok(die_coarse_create_runtime_v2(model, &runtime));
    for (unsigned i = 0; i < data.at("execution_count").get<unsigned>(); ++i) {
      const auto fair_owner = i / 2 + 1;
      const DieOwnerV2 who{sizeof(DieOwnerV2), 2, i + 1, fair_owner, fair_owner, 1};
      DieHandle execution = 0;
      ok(die_coarse_submit_plan_v2(runtime,
          compiled.at("results").at(i % 2).at("plan_handle").get<DieHandle>(), &who, 0, &execution));
    }
    for (const auto& tick : data.at("event_ticks_ns")) {
      require(next_event(runtime) == tick, "real compiled fixture reaches its expected public event");
      ok(die_coarse_advance_to_v2(runtime, tick.get<std::uint64_t>()));
    }
    return runtime;
  };
  const auto source = make_world();
  const auto saved = snapshot(source);
  // Each epoch of these actual Plans has one activity: two split executions
  // therefore use at most two slots per fair owner at every internal phase.
  std::map<std::string, unsigned> owner_executions;
  for (const auto& execution : saved.at("executions")) {
    ++owner_executions[execution.at("owner").at("key").get<std::string>()];
    for (const auto& epoch : execution.at("plan").at("epochs"))
      require(epoch.at("activities").size() == 1, "actual Plan uses one activity per epoch");
  }
  require(owner_executions.size() == 4, "eight executions have four fair owners");
  for (const auto& entry : owner_executions)
    require(entry.second == 2, "each fair owner owns two executions and at most two slots");
  DieHandle restored = 0, fork = 0;
  std::string restore_errors;
  try { restored = imported(model, source, saved); }
  catch (const std::exception& error) { restore_errors += std::string(error.what()) + "\n"; }
  try { fork = forked(source); }
  catch (const std::exception& error) { restore_errors += std::string(error.what()) + "\n"; }
  require(restore_errors.empty(), restore_errors);
  auto bad = saved;
  const auto first_active = std::size_t{1};
  require(saved.at("executions").at(0).at("activities").at(first_active).at("rate").get<double>() > 0,
          "real first-work activity is running");
  const auto invalid_rate = [&](double rate, const char* label) {
    auto edited = saved;
    edited["executions"][0]["activities"][first_active]["rate"] = rate;
    reject(model, source, edited, label);
  };
  const double rate = saved.at("executions").at(0).at("activities").at(first_active).at("rate");
  invalid_rate(rate * 1.01, "increased rate");
  invalid_rate(0, "structural-zero replacement");
  double low = rate;
  for (unsigned i = 0; i < 4; ++i) low = std::nextafter(low, 0.0);
  invalid_rate(low, "merely feasible rate beyond one ULP");
  bad["executions"][0]["activities"][first_active]["alpha"] = .5;
  reject(model, source, bad, "changed owner alpha");
  bad = saved; bad["executions"][0]["activities"][0]["rate"] = std::numeric_limits<double>::denorm_min();
  reject(model, source, bad, "inactive structural zero");
  bad = saved; bad["executions"][0]["credit"][0] = 1;
  reject(model, source, bad, "credit inconsistent with work");
  bad = saved; bad["capacities"][0] = 2;
  reject(model, source, bad, "capacity above hardware peak", DIE_INVALID_HARDWARE);
  bad = saved; bad["hardware_fingerprint"] = "invalid";
  reject(model, source, bad, "wrong hardware fingerprint", DIE_INCOMPATIBLE_FINGERPRINT);
  auto adjacent = saved;
  adjacent["executions"][0]["activities"][first_active]["rate"] = std::nextafter(rate, 0.0);
  const auto inward = imported(model, source, adjacent);
  compare_and_drain({inward});
  ok(die_coarse_destroy_runtime_v2(inward));

  // A separately submitted source supplies a reference unaffected by import.
  const auto controlled = make_world();
  require(snapshot(controlled) == saved, "independent source recreates the original world");
  const auto controlled_import = imported(model, controlled, saved);
  const auto controlled_fork = forked(controlled);
  const std::vector<DieHandle> changed{controlled, controlled_import, controlled_fork};
  for (auto world : changed) {
    const auto listed = take(die_coarse_list_executions_v2(world)).at("executions");
    require(listed.size() == 8, "all split executions retain their identities");
    DieHandle last = 0;
    for (const auto& execution : listed)
      if (execution.at("local_handle") == 8) last = execution.at("execution_handle");
    require(last != 0, "last split execution remains addressable");
    ok(die_coarse_cancel_execution_v2(last, 1165));
    ok(die_coarse_update_capacity_v2(world, HBM0,
        saved.at("capacities").at(HBM0).get<double>() * .75, 1165));
  }
  compare_and_drain({source, restored, fork});
  for (auto world : {source, restored, fork}) {
    const auto terminal = snapshot(world).at("executions");
    require(terminal.size() == 8, "uncontrolled continuation keeps all eight executions");
    for (const auto& execution : terminal)
      require(execution.at("terminal").get<bool>() && !execution.at("cancelled").get<bool>(),
              "uncontrolled source/import/fork completes all eight executions successfully");
  }
  compare_and_drain(changed);
  const auto done = snapshot(controlled);
  unsigned completed = 0, cancelled = 0;
  for (const auto& execution : done.at("executions")) {
    require(execution.at("terminal").get<bool>(), "controlled continuation reaches every terminal");
    execution.at("cancelled").get<bool>() ? ++cancelled : ++completed;
  }
  require(completed == 7 && cancelled == 1, "only the explicitly cancelled execution is cancelled");
  for (auto world : std::vector<DieHandle>{source, restored, fork, controlled, controlled_import, controlled_fork})
    ok(die_coarse_destroy_runtime_v2(world));
  for (const auto& plan : compiled.at("results"))
    ok(die_coarse_release_plan_v2(plan.at("plan_handle").get<DieHandle>()));
  ok(die_coarse_destroy_model_v2(model));
  check_outward_tangent();
}
}
int main() { return run_test("runtime_snapshot_split_owner_test", verify); }
