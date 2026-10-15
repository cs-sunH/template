// Production lifecycle regression for the explicit GEMM OS K-readiness
// program. The existing diagnostic driver supplies the independent explicit
// slot reference and the real per-K resource facts; this test exercises the
// public compiled Plan and Runtime path.
#define main k_readiness_runtime_driver_entry
#include "k_readiness_runtime_driver.cc"
#undef main
#include "test_support.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using nlohmann::json;
using coarse_die::Hardware;
using coarse_die::Owner;
using coarse_die::Plan;
using coarse_die::Runtime;
using coarse_die::compile;
using coarse_die::plan_from_json;
using coarse_die::plan_to_json;
using coarse_die_test::require;

constexpr std::uint64_t kMaxExpandedWaves = 4096;
constexpr std::uint64_t kArrivalOffsetNs = 50000000ULL;
constexpr std::size_t kEventLimit = 1000000;

Shape full_tile_shape() { return {80, 128, 4096}; }
Shape tail_shape() { return {65, 65, 4097}; }
Shape small_shape() { return {16, 64, 64}; }

json task_for(const Shape& shape, bool program) {
  json task = gemm_task(shape);
  if (program) {
    task["runtime_program"] = {
        {"kind", "gemm_os_k_readiness_v1"},
        {"max_expanded_waves", kMaxExpandedWaves}};
  }
  return task;
}

std::shared_ptr<const Plan> compile_plan(const Hardware& hardware,
                                         const Shape& shape,
                                         bool program) {
  return compile(hardware, task_for(shape, program));
}

void require_u64_equal(const json& actual, const json& expected,
                       const std::string& label) {
  require(actual.is_number_unsigned() || actual.is_number_integer(),
          label + " must be an integer");
  require(expected.is_number_unsigned() || expected.is_number_integer(),
          label + " reference must be an integer");
  require(actual.get<std::uint64_t>() == expected.get<std::uint64_t>(),
          label + " differs from the explicit-slot reference");
}

struct ProgramShape {
  std::uint64_t waves = 0;
  std::uint64_t outputs = 0;
  std::uint64_t k_tiles = 0;
};

ProgramShape validate_compiled_program(const Hardware& hardware,
                                       const Plan& plan) {
  require(plan.epochs.empty(), "program Plans must leave epochs empty");
  require(plan.gemm_os_program.has_value(),
          "explicit opt-in must compile to a GEMM OS program Plan");
  const json wire = plan_to_json(plan);
  require(wire.at("schema") == "coarse_die_plan_v4",
          "program Plans use the v4 wire schema");
  require(wire.at("program").at("kind") == "gemm_os_k_readiness_v1" &&
              wire.at("program").at("max_expanded_waves") ==
                  plan.gemm_os_program->max_expanded_waves,
          "program Plan wire preserves its opt-in kind and expansion bound");
  require(wire.at("fingerprint") == plan.fingerprint,
          "program Plan fingerprint must cover its serialized program option");
  require(plan_to_json(*plan_from_json(wire)) == wire,
          "program Plan v4 wire roundtrips canonically");

  k_readiness::Kernel kernel(
      hardware, k_readiness::geometry_input(hardware, plan), false);
  const ProgramShape shape{kernel.wave_count(), kernel.output_count(),
                           kernel.geometry().k_tiles()};
  require(shape.waves > 0 && shape.outputs > 0 && shape.k_tiles > 0,
          "compiled program geometry must be nonempty");
  require(shape.waves <= kMaxExpandedWaves,
          "test GEMM expansion stays within its declared bound");
  require(shape.waves == shape.outputs * shape.k_tiles,
          "reference geometry must partition waves by output tile and K tile");
  return shape;
}

void validate_program_progress(const json& progress,
                               const ProgramShape& expected_shape) {
  require(progress.contains("program") && progress.at("program").is_object(),
          "program execution exposes its readiness state");
  const json& program = progress.at("program");
  require(program.at("kind") == "gemm_os_k_readiness_v1",
          "progress identifies the opt-in program kind");
  require_u64_equal(program.at("wave_count"), expected_shape.waves,
                    "program wave count");
  require_u64_equal(program.at("output_count"), expected_shape.outputs,
                    "program output count");

  const json& cursor = program.at("cursor");
  const std::uint64_t load_started = cursor.at("load_started").get<std::uint64_t>();
  const std::uint64_t load_completed = cursor.at("load_completed").get<std::uint64_t>();
  const std::uint64_t compute_started = cursor.at("compute_started").get<std::uint64_t>();
  const std::uint64_t compute_completed = cursor.at("compute_completed").get<std::uint64_t>();
  const std::uint64_t store_started = cursor.at("store_started").get<std::uint64_t>();
  const std::uint64_t store_completed = cursor.at("store_completed").get<std::uint64_t>();
  require(load_completed <= load_started && load_started <= expected_shape.waves,
          "Load cursor counters stay ordered and in range");
  require(compute_started <= load_completed &&
              compute_completed <= compute_started &&
              compute_started <= expected_shape.waves,
          "Compute starts only after Load terminal and counters stay ordered");
  require(store_started <= store_completed + 1 &&
              store_completed <= store_started &&
              store_started <= expected_shape.outputs,
          "Store cursor counters stay ordered and in range");

  const auto input_occupancy = program.at("input_occupancy").get<std::uint64_t>();
  const auto output_occupancy = program.at("output_occupancy").get<std::uint64_t>();
  require(input_occupancy <= 2, "program never exceeds two input slots");
  require(output_occupancy <= 2, "program never exceeds two output slots");

  require(program.at("active_waves").is_object(),
          "program reports the active stage waves");
  const json& active = program.at("active_waves");
  const json& activities = progress.at("activities");
  require(activities.is_array() && activities.size() <= 3,
          "program retains at most one active Load, Compute, and Store");
  static constexpr const char* roles[] = {"load", "compute", "store"};
  std::size_t role_cursor = 0;
  for (const auto& activity : activities) {
    const auto role = activity.at("role").get<std::string>();
    while (role_cursor < std::size(roles) && role != roles[role_cursor])
      ++role_cursor;
    require(role_cursor < std::size(roles),
            "active stage activities are in Load/Compute/Store order");
    const json& ordinal = activity.at("wave_ordinal");
    require(!ordinal.is_null(), "active program activity has a wave ordinal");
    require(active.at(role) == ordinal,
            "active wave metadata matches its live activity");
    ++role_cursor;
  }
  for (const char* role : roles) {
    if (active.at(role).is_null()) continue;
    require(std::any_of(activities.begin(), activities.end(), [&](const json& item) {
              return item.at("role") == role &&
                     item.at("wave_ordinal") == active.at(role);
            }),
            std::string("active wave has a live ") + role + " activity");
  }
  require(program.at("output_full_nonfinal_compute_observations").is_number_integer() ||
              program.at("output_full_nonfinal_compute_observations").is_number_unsigned(),
          "program reports nonfinal Compute readiness observations");
  require(program.at("final_compute_output_full_waits").is_number_integer() ||
              program.at("final_compute_output_full_waits").is_number_unsigned(),
          "program reports final Compute output-slot waits");
}

struct ProgramRun {
  std::vector<json> progress;
  std::vector<std::uint64_t> finishes;
  json counters;
  std::size_t event_advances = 0;
};

ProgramRun run_programs(const Hardware& hardware,
                        std::shared_ptr<const Plan> plan,
                        const ProgramShape& shape,
                        const std::vector<std::uint64_t>& arrivals) {
  require(!arrivals.empty(), "program run needs at least one arrival");
  Runtime runtime(hardware);
  std::vector<std::uint64_t> handles;
  handles.reserve(arrivals.size());
  for (std::size_t i = 0; i < arrivals.size(); ++i) {
    if (i != 0) runtime.advance_to(arrivals[i]);
  handles.push_back(runtime.submit(plan, owner(static_cast<std::uint64_t>(i + 1)),
                                     arrivals[i]));
    validate_program_progress(runtime.query_progress(handles.back()), shape);
  }

  ProgramRun result;
  for (;;) {
    bool all_terminal = true;
    for (const auto handle : handles) {
      const auto progress = runtime.query_progress(handle);
      validate_program_progress(progress, shape);
      if (!progress.at("terminal").get<bool>()) all_terminal = false;
    }
    if (all_terminal) break;
    const auto next = runtime.next_event();
    require(next.has_value(), "live program work has a next event");
    require(*next > runtime.now_ns(), "program next event advances time");
    runtime.advance_to(*next);
    ++result.event_advances;
    require(result.event_advances <= kEventLimit,
            "test event loop remains within the fixed regression bound");
  }

  require(runtime.drained(), "all program executions reached terminal state");
  for (const auto handle : handles) {
    const auto progress = runtime.query_progress(handle);
    validate_program_progress(progress, shape);
    require(progress.at("state") == "finished" &&
                !progress.at("cancelled").get<bool>(),
            "successful GEMM program finishes without cancellation");
    const json& program = progress.at("program");
    const json& cursor = program.at("cursor");
    require(cursor.at("load_started") == shape.waves &&
                cursor.at("load_completed") == shape.waves &&
                cursor.at("compute_started") == shape.waves &&
                cursor.at("compute_completed") == shape.waves &&
                cursor.at("store_started") == shape.outputs &&
                cursor.at("store_completed") == shape.outputs,
            "successful program conserves every Load, Compute, and final Store");
    require(progress.at("activities").empty(),
            "successful program releases all live stage activities");
    for (const char* role : {"load", "compute", "store"})
      require(program.at("active_waves").at(role).is_null(),
              std::string("successful program has no active ") + role + " wave");
    result.finishes.push_back(progress.at("finish_ns").get<std::uint64_t>());
    result.progress.push_back(progress);
  }
  result.counters = runtime.counters();
  return result;
}

json run_explicit_slot_reference(const Hardware& hardware,
                                 const Plan& program_plan,
                                 const std::vector<std::uint64_t>& arrivals) {
  k_readiness::Kernel kernel(
      hardware, k_readiness::geometry_input(hardware, program_plan), false);
  k_readiness::Driver driver(hardware, kernel, arrivals);
  return driver.run();
}

json reference_counters(const json& report) {
  return report.at("counters");
}

std::uint64_t reference_finish(const json& report, std::size_t request) {
  if (report.contains("finish_ns"))
    return report.at("finish_ns").get<std::uint64_t>();
  return report.at("finish_ns_by_request")
      .at("comp" + std::to_string(request)).get<std::uint64_t>();
}

void require_relative_error(long double actual, long double expected,
                            long double maximum, const std::string& label) {
  if (expected == 0.0L) {
    require(actual == 0.0L, label + " must remain exactly zero");
    return;
  }
  const long double relative = std::abs(actual - expected) / std::abs(expected);
  require(relative <= maximum,
          label + " exceeds its frozen relative-error gate: actual=" +
              std::to_string(actual) + " expected=" + std::to_string(expected) +
              " relative_error=" + std::to_string(relative) +
              " limit=" + std::to_string(maximum));
}

void compare_concurrent_busy(const json& actual, const json& expected,
                             const std::string& label) {
  constexpr long double kConcurrentBusyGate = 0.05L;
  for (const char* field : {"hbm_busy_ns", "hbm_shared_ns", "compute_busy_ns"}) {
    require_relative_error(actual.at(field).get<long double>(),
                           expected.at(field).get<long double>(),
                           kConcurrentBusyGate,
                           label + " " + field);
  }
  for (const char* field : {"row_busy_ns", "row_shared_ns"}) {
    for (std::size_t row = 0; row < 2; ++row) {
      require_relative_error(actual.at(field).at(row).get<long double>(),
                             expected.at(field).at(row).get<long double>(),
                             kConcurrentBusyGate,
                             label + " " + field + " row " +
                                 std::to_string(row));
    }
  }
}

void compare_work_conservation(const json& actual, const json& expected,
                               const std::string& label) {
  const char* actual_hbm_key = actual.contains("hbm_served_bytes_by_row_direction_exact")
      ? "hbm_served_bytes_by_row_direction_exact"
      : "hbm_bytes_by_row_direction_exact";
  const char* expected_hbm_key = expected.contains("hbm_served_bytes_by_row_direction_exact")
      ? "hbm_served_bytes_by_row_direction_exact"
      : "hbm_bytes_by_row_direction_exact";
  require(actual.at(actual_hbm_key) == expected.at(expected_hbm_key),
          label + " exact HBM payload differs from explicit-slot reference: actual=" +
              actual.at(actual_hbm_key).dump() + " expected=" +
              expected.at(expected_hbm_key).dump());
  const json& actual_comp = actual.contains("served_by_category")
      ? actual.at("served_by_category").at("comp") : actual;
  const json& expected_comp = expected.contains("served_by_category")
      ? expected.at("served_by_category").at("comp") : expected;
  require(actual_comp.at("int4_ops_exact_decimal") ==
              expected_comp.at("int4_ops_exact_decimal"),
          label + " exact INT4 work differs from explicit-slot reference: actual=" +
              actual_comp.at("int4_ops_exact_decimal").dump() + " expected=" +
              expected_comp.at("int4_ops_exact_decimal").dump());
  require(actual_comp.at("vec_eqops_exact_decimal") ==
              expected_comp.at("vec_eqops_exact_decimal"),
          label + " exact vector-equivalent work differs from explicit-slot reference: actual=" +
              actual_comp.at("vec_eqops_exact_decimal").dump() + " expected=" +
              expected_comp.at("vec_eqops_exact_decimal").dump());
}

void compare_program_run(const ProgramRun& actual, const json& expected,
                         const json& expected_per_request,
                         const std::vector<std::uint64_t>& arrivals,
                         const std::string& label) {
  require(actual.finishes.size() == arrivals.size(),
          label + " returns one finish time per request");
  require(actual.progress.size() == arrivals.size(),
          label + " returns one final progress record per request");
  const long double finish_gate = arrivals.size() == 1 ? 0.03L : 0.05L;
  for (std::size_t i = 0; i < arrivals.size(); ++i) {
    require_relative_error(
        static_cast<long double>(actual.finishes[i]),
        static_cast<long double>(reference_finish(expected, i)), finish_gate,
        label + " finish_ns request " + std::to_string(i));
    compare_work_conservation(actual.progress[i].at("served"),
                              reference_counters(expected_per_request),
                              label + " request " + std::to_string(i));
  }
  compare_work_conservation(actual.counters,
                            reference_counters(expected), label + " world totals");
  if (arrivals.size() > 1)
    compare_concurrent_busy(actual.counters, reference_counters(expected), label);
}

void compiled_solo_and_arrival_work_conservation() {
  const Hardware hardware = synthetic_hardware();

  const Shape full = full_tile_shape();
  const auto full_plan = compile_plan(hardware, full, true);
  const ProgramShape full_geometry = validate_compiled_program(hardware, *full_plan);
  const auto full_reference = run_explicit_slot_reference(
      hardware, *full_plan, {0});
  const auto full_actual = run_programs(
      hardware, full_plan, full_geometry, {0});
  require(full_geometry.waves == 640,
          "full-tile fixture retains the previously selected 640-wave geometry");
  compare_program_run(full_actual, full_reference, full_reference,
                      {0}, "full-tile solo");

  const Shape tail = tail_shape();
  const auto tail_plan = compile_plan(hardware, tail, true);
  const ProgramShape tail_geometry = validate_compiled_program(hardware, *tail_plan);
  require(tail_geometry.waves == 650,
          "M/N/K-tail fixture retains the previously selected 650-wave geometry");
  const auto tail_reference = run_explicit_slot_reference(
      hardware, *tail_plan, {0, kArrivalOffsetNs});
  const auto tail_solo_reference = run_explicit_slot_reference(
      hardware, *tail_plan, {0});
  const auto tail_actual = run_programs(
      hardware, tail_plan, tail_geometry, {0, kArrivalOffsetNs});
  compare_program_run(tail_actual, tail_reference, tail_solo_reference,
                      {0, kArrivalOffsetNs}, "M/N/K-tail offset arrival");
}

void ready_launch_and_terminal_wins_same_tick_cancel() {
  const Hardware hardware = synthetic_hardware();
  const Shape shape = small_shape();
  auto compiled = compile_plan(hardware, shape, true);
  const ProgramShape geometry = validate_compiled_program(hardware, *compiled);

  const json estimate = coarse_die::estimate_alone(hardware, *compiled);
  require(estimate.at("plan_schema") == "coarse_die_plan_v4" &&
              estimate.at("compiled_program_plan_executed") == true &&
              estimate.at("measurement_source") ==
                  "diagnostic_cpp_k_ready_facts_and_actual_public_runtime",
          "estimate_alone identifies the compiled v4 program it actually executed");
  require(estimate.at("domain_work").is_array() &&
              estimate.at("domain_work").size() == coarse_die::resource_count,
          "program estimate publishes work for all 13 Runtime domains");
  k_readiness::Kernel estimate_facts(
      hardware, k_readiness::geometry_input(hardware, *compiled), false);
  std::array<bool, coarse_die::resource_count> resource_used{};
  for (std::uint64_t wave = 0; wave < estimate_facts.wave_count(); ++wave) {
    for (const auto& fact : estimate_facts.descriptor().facts(wave)) {
      for (std::size_t resource = 0; resource < coarse_die::resource_count;
           ++resource) {
        resource_used[resource] = resource_used[resource] ||
                                  fact.work[resource] > 0.0;
      }
    }
  }
  for (std::size_t resource = 0; resource < coarse_die::resource_count;
       ++resource) {
    const bool estimated_used = estimate.at("domain_work").at(resource)
                                    .get<double>() > 0.0;
    require(estimated_used == resource_used[resource],
            "program estimate preserves every used physical resource domain");
  }
  require(estimate.at("execution").at("terminal") == true &&
              estimate.at("execution").at("program").at("cursor")
                      .at("store_completed") == geometry.outputs,
          "estimate_alone returns the completed discrete program execution");

  auto delayed = std::make_shared<Plan>(*compiled);
  delayed->launch_ns = 1234567;
  delayed->fingerprint.clear();
  delayed->fingerprint = plan_to_json(*delayed).at("fingerprint").get<std::string>();
  Runtime launch_runtime(hardware);
  constexpr std::uint64_t issue_ns = 321;
  const auto delayed_handle = launch_runtime.submit(
      delayed, owner(10), issue_ns);
  auto delayed_progress = launch_runtime.query_progress(delayed_handle);
  require(delayed_progress.at("ready_ns") == issue_ns + delayed->launch_ns,
          "program ready time includes the Plan launch delay");
  validate_program_progress(delayed_progress, geometry);
  require(delayed_progress.at("activities").empty(),
          "program has no stage activity before launch readiness");
  require(delayed_progress.at("program").at("cursor").at("load_started") == 0,
          "Load cannot start before launch readiness");
  launch_runtime.advance_to(issue_ns + delayed->launch_ns - 1);
  delayed_progress = launch_runtime.query_progress(delayed_handle);
  require(delayed_progress.at("activities").empty() &&
              delayed_progress.at("program").at("cursor").at("load_started") == 0,
          "program stays queued through the nanosecond before readiness");
  launch_runtime.advance_to(issue_ns + delayed->launch_ns);
  delayed_progress = launch_runtime.query_progress(delayed_handle);
  validate_program_progress(delayed_progress, geometry);
  require(delayed_progress.at("program").at("cursor").at("load_started") > 0,
          "first program wave becomes eligible at its ready tick");

  const json reference = run_explicit_slot_reference(hardware, *compiled, {0});
  const auto expected_finish = reference_finish(reference, 0);
  require_relative_error(
      estimate.at("estimated_latency_ns").get<long double>(),
      static_cast<long double>(expected_finish), 0.03L,
      "estimate_alone small full-tile finish_ns");
  Runtime terminal_runtime(hardware);
  const auto terminal_handle = terminal_runtime.submit(
      compiled, owner(11), 0);
  terminal_runtime.cancel(terminal_handle, expected_finish);
  const auto terminal_progress = terminal_runtime.query_progress(terminal_handle);
  require(terminal_progress.at("state") == "finished" &&
              !terminal_progress.at("cancelled").get<bool>() &&
              terminal_progress.at("finish_ns") == expected_finish,
          "same-tick final Store terminal wins over cancel");
  const json& cursor = terminal_progress.at("program").at("cursor");
  require(cursor.at("store_completed") == geometry.outputs,
          "same-tick terminal records the final Store completion");
}

void partial_cancel_preserves_only_completed_cursor_and_served_work() {
  const Hardware hardware = synthetic_hardware();
  const Shape shape = tail_shape();
  const auto plan = compile_plan(hardware, shape, true);
  const ProgramShape geometry = validate_compiled_program(hardware, *plan);
  Runtime runtime(hardware);
  const auto handle = runtime.submit(plan, owner(20), 0);
  const auto next = runtime.next_event();
  require(next.has_value() && *next > 1,
          "tail program first service interval is long enough for a partial cancel");
  const std::uint64_t cancel_ns = *next / 2;
  require(cancel_ns > runtime.now_ns() && cancel_ns < *next,
          "partial cancellation tick precedes the first stage terminal");
  runtime.advance_to(cancel_ns);
  const auto before_cancel = runtime.query_progress(handle);
  validate_program_progress(before_cancel, geometry);
  require(!before_cancel.at("activities").empty(),
          "partial program cancellation retains a live stage");
  const bool partial_work = std::any_of(
      before_cancel.at("activities").begin(), before_cancel.at("activities").end(),
      [](const json& activity) {
        const double progress = activity.at("progress").get<double>();
        return progress > 0.0 && progress < 1.0;
      });
  require(partial_work, "cancel fixture reaches a partial stage progress");
  const json cursor_before = before_cancel.at("program").at("cursor");
  runtime.cancel(handle, cancel_ns);
  const auto cancelled = runtime.query_progress(handle);
  require(cancelled.at("state") == "cancelled" &&
              cancelled.at("cancelled").get<bool>() &&
              cancelled.at("finish_ns") == cancel_ns,
          "partial cancellation records its requested absolute tick");
  require(cancelled.at("program").at("cursor") == cursor_before,
          "partial cancellation does not invent a stage terminal");
  for (const auto& activity : cancelled.at("activities"))
    require(activity.at("rate") == 0.0,
            "cancelled active program stages have zero rate");
  require(cancelled.at("served").at("hbm_bytes_by_row_direction_exact") !=
              json::array({{"0", "0"}, {"0", "0"}}),
          "partial cancellation retains the HBM payload served before cancellation");
}

void capacity_pause_and_resume() {
  const Hardware hardware = synthetic_hardware();
  const Shape shape = small_shape();
  const auto plan = compile_plan(hardware, shape, true);
  const ProgramShape geometry = validate_compiled_program(hardware, *plan);
  const json reference = run_explicit_slot_reference(hardware, *plan, {0});

  Runtime runtime(hardware);
  for (std::size_t resource = 0; resource < coarse_die::resource_count; ++resource) {
    if (hardware.rates[resource] > 0.0)
      runtime.update_capacity(resource, 0.0, 0);
  }
  const auto handle = runtime.submit(plan, owner(30), 0);
  auto paused = runtime.query_progress(handle);
  validate_program_progress(paused, geometry);
  const json paused_cursor = paused.at("program").at("cursor");
  for (const auto& activity : paused.at("activities"))
    require(activity.at("rate") == 0.0,
            "zero physical capacity pauses every live program stage");
  require(!runtime.next_event().has_value(),
          "zero physical capacity creates no synthetic completion event");
  runtime.advance_to(5000000ULL);
  paused = runtime.query_progress(handle);
  require(paused.at("program").at("cursor") == paused_cursor,
          "capacity pause does not advance stage counters");

  for (std::size_t resource = 0; resource < coarse_die::resource_count; ++resource) {
    if (hardware.rates[resource] > 0.0)
      runtime.update_capacity(resource, hardware.rates[resource], runtime.now_ns());
  }
  std::size_t events = 0;
  while (!runtime.terminal_finish_ns(handle)) {
    const auto next = runtime.next_event();
    require(next.has_value() && *next > runtime.now_ns(),
            "capacity recovery restores a finite next event");
    runtime.advance_to(*next);
    validate_program_progress(runtime.query_progress(handle), geometry);
    require(++events <= kEventLimit,
            "capacity recovery stays within the fixed event budget");
  }
  const auto final = runtime.query_progress(handle);
  require(final.at("state") == "finished" &&
              final.at("program").at("cursor").at("store_completed") == geometry.outputs,
          "capacity recovery drains every output tile");
  compare_work_conservation(runtime.counters(), reference_counters(reference),
                            "capacity pause/resume");
  require(final.at("served").at("hbm_bytes_by_row_direction_exact") ==
              reference_counters(reference).at("hbm_served_bytes_by_row_direction_exact"),
          "capacity recovery conserves exact HBM payload");
}

enum class ProgramSnapshotPoint {
  NonfinalComputeCompleted,
  FirstStoreActive,
  StoresCompleted
};

void require_program_snapshot_roundtrip(const Hardware& hardware,
                                        std::shared_ptr<const Plan> plan,
                                        const ProgramShape& geometry,
                                        ProgramSnapshotPoint point,
                                        std::uint64_t store_target,
                                        const std::string& label) {
  Runtime source(hardware);
  const auto handle = source.submit(plan, owner(35), 0);
  bool reached = false;
  std::size_t events = 0;
  while (!reached) {
    const json progress = source.query_progress(handle);
    validate_program_progress(progress, geometry);
    const json& cursor = progress.at("program").at("cursor");
    const auto compute_completed = cursor.at("compute_completed").get<std::uint64_t>();
    const auto store_started = cursor.at("store_started").get<std::uint64_t>();
    const auto store_completed = cursor.at("store_completed").get<std::uint64_t>();
    switch (point) {
      case ProgramSnapshotPoint::NonfinalComputeCompleted:
        reached = compute_completed > 0 &&
                  compute_completed < geometry.k_tiles && store_started == 0;
        break;
      case ProgramSnapshotPoint::FirstStoreActive:
        reached = store_started == 1 && store_completed == 0;
        break;
      case ProgramSnapshotPoint::StoresCompleted:
        reached = store_completed == store_target;
        break;
    }
    if (reached) break;
    require(!progress.at("terminal").get<bool>(),
            label + " target state occurs before whole-program terminal");
    const auto next = source.next_event();
    require(next.has_value() && *next > source.now_ns(),
            label + " has a finite advancing event");
    source.advance_to(*next);
    require(++events <= kEventLimit,
            label + " remains within the fixed event budget");
  }
  if (point == ProgramSnapshotPoint::FirstStoreActive) {
    const auto progress = source.query_progress(handle);
    require(std::any_of(progress.at("activities").begin(),
                        progress.at("activities").end(),
                        [](const json& activity) {
                          return activity.at("role") == "store";
                        }),
            label + " captures the final output while Store is live");
  }

  const json snapshot = source.export_snapshot();
  require(snapshot.at("schema") == "coarse_die_snapshot_v4",
          label + " uses the program snapshot wire");
  const json& saved_program = snapshot.at("executions").at(0).at("program");
  const json queried_progress = source.query_progress(handle);
  const json& queried_program = queried_progress.at("program");
  require(saved_program.at("kind") == queried_program.at("kind") &&
              saved_program.at("cursor") == queried_program.at("cursor") &&
              saved_program.at("output_full_nonfinal_compute_observations") ==
                  queried_program.at("output_full_nonfinal_compute_observations") &&
              saved_program.at("final_compute_output_full_waits") ==
                  queried_program.at("final_compute_output_full_waits") &&
              saved_program.contains("last_final_wait_wave"),
          label + " snapshots the cursor and readiness diagnostics");
  auto imported = Runtime::import_snapshot(hardware, snapshot);
  require(imported->export_snapshot() == snapshot,
          label + " imports and re-exports with exact snapshot equality");
  require(imported->query_progress(handle) == source.query_progress(handle) &&
              imported->counters() == source.counters(),
          label + " restores exact progress and cumulative work totals");
}

void program_snapshot_prefix_boundaries() {
  const Hardware hardware = synthetic_hardware();
  const Shape full_shape = full_tile_shape();
  const auto full_plan = compile_plan(hardware, full_shape, true);
  const ProgramShape full_geometry = validate_compiled_program(hardware, *full_plan);
  require_program_snapshot_roundtrip(
      hardware, full_plan, full_geometry,
      ProgramSnapshotPoint::NonfinalComputeCompleted, 0,
      "full-tile nonfinal Compute prefix");
  for (const std::uint64_t stored : {1ULL, 5ULL}) {
    require_program_snapshot_roundtrip(
        hardware, full_plan, full_geometry,
        ProgramSnapshotPoint::StoresCompleted, stored,
        "full-tile Store prefix " + std::to_string(stored));
  }

  const Shape tail = tail_shape();
  const auto tail_plan = compile_plan(hardware, tail, true);
  const ProgramShape tail_geometry = validate_compiled_program(hardware, *tail_plan);
  require_program_snapshot_roundtrip(
      hardware, tail_plan, tail_geometry,
      ProgramSnapshotPoint::NonfinalComputeCompleted, 0,
      "tail nonfinal Compute prefix");
  require_program_snapshot_roundtrip(
      hardware, tail_plan, tail_geometry,
      ProgramSnapshotPoint::FirstStoreActive, 0,
      "tail final output reservation");
  for (const std::uint64_t stored : {1ULL, 5ULL, 9ULL}) {
    require_program_snapshot_roundtrip(
        hardware, tail_plan, tail_geometry,
        ProgramSnapshotPoint::StoresCompleted, stored,
        "tail Store prefix " + std::to_string(stored));
  }
}

void v4_program_snapshot_import_fork_and_atomic_rejection() {
  const Hardware hardware = synthetic_hardware();
  const Shape shape = tail_shape();
  const auto plan = compile_plan(hardware, shape, true);
  const ProgramShape geometry = validate_compiled_program(hardware, *plan);
  Runtime source(hardware);
  const auto handle = source.submit(plan, owner(40), 0);

  bool partial_progress = false;
  for (std::size_t event = 0; event < 8 && !partial_progress; ++event) {
    const auto next = source.next_event();
    require(next.has_value() && *next > source.now_ns(),
            "snapshot source has an advancing event");
    const auto midpoint = source.now_ns() + std::max<std::uint64_t>(
        1, (*next - source.now_ns()) / 2);
    source.advance_to(midpoint);
    const auto progress = source.query_progress(handle);
    validate_program_progress(progress, geometry);
    partial_progress = std::any_of(
        progress.at("activities").begin(), progress.at("activities").end(),
        [](const json& activity) {
          const double value = activity.at("progress").get<double>();
          return value > 0.0 && value < 1.0;
        });
  }
  require(partial_progress,
          "snapshot fixture captures fractional progress in a live K stage");

  const json snapshot = source.export_snapshot();
  require(snapshot.at("schema") == "coarse_die_snapshot_v4" &&
              snapshot.at("snapshot_version") == 4,
          "program Runtime exports snapshot v4");
  const json& execution_wire = snapshot.at("executions").at(0);
  require(execution_wire.contains("program") &&
              execution_wire.at("program").at("cursor") ==
                  source.query_progress(handle).at("program").at("cursor"),
          "snapshot stores the same program cursor as query_progress");
  for (const auto& activity : execution_wire.at("activities"))
    require(activity.contains("wave_ordinal"),
            "snapshot identifies each active stage's wave ordinal");

  auto imported = Runtime::import_snapshot(hardware, snapshot);
  auto fork = Runtime::import_snapshot(hardware, imported->export_snapshot());
  require(imported->export_snapshot() == snapshot &&
              fork->export_snapshot() == snapshot,
          "program snapshot source/import/fork begin from identical wire state");
  require(imported->query_progress(handle) == source.query_progress(handle) &&
              fork->query_progress(handle) == source.query_progress(handle) &&
              imported->counters() == source.counters() &&
              fork->counters() == source.counters(),
          "program snapshot roundtrip preserves progress and cumulative accounting");

  json invalid = snapshot;
  const auto load_started = invalid.at("executions").at(0)
                                .at("program").at("cursor")
                                .at("load_started").get<std::uint64_t>();
  invalid["executions"][0]["program"]["cursor"]["load_completed"] =
      load_started + 1;
  bool rejected = false;
  try {
    (void)Runtime::import_snapshot(hardware, invalid);
  } catch (const std::exception& error) {
    rejected = std::string(error.what()).find("invalid_snapshot") !=
               std::string::npos;
  }
  require(rejected, "import rejects a program cursor with impossible completion");
  require(source.export_snapshot() == snapshot,
          "failed program import leaves the source Runtime untouched");

  std::size_t steps = 0;
  while (!source.drained()) {
    const auto next = source.next_event();
    require(next.has_value() && *next > source.now_ns(),
            "snapshot continuation has a finite next event");
    require(imported->next_event() == next && fork->next_event() == next,
            "snapshot source/import/fork choose the same next event");
    source.advance_to(*next);
    imported->advance_to(*next);
    fork->advance_to(*next);
    const json source_progress = source.query_progress(handle);
    validate_program_progress(source_progress, geometry);
    require(imported->query_progress(handle) == source_progress &&
                fork->query_progress(handle) == source_progress,
            "snapshot source/import/fork progress continues strictly equal");
    require(imported->counters() == source.counters() &&
                fork->counters() == source.counters() &&
                imported->export_snapshot() == source.export_snapshot() &&
                fork->export_snapshot() == source.export_snapshot(),
            "snapshot source/import/fork accounting and wire state continue strictly equal");
    require(++steps <= kEventLimit,
            "snapshot continuation stays within the fixed event budget");
  }
  require(source.drained() && imported->drained() && fork->drained(),
          "all three snapshot continuations drain together");
}

void default_plan_and_snapshot_keep_legacy_wire() {
  const Hardware hardware = synthetic_hardware();
  const Shape shape = small_shape();
  const auto legacy_plan = compile_plan(hardware, shape, false);
  const json plan_wire = plan_to_json(*legacy_plan);
  require(!legacy_plan->gemm_os_program.has_value(),
          "default compiler Plans do not enable the opt-in program");
  require(plan_wire.at("schema") == "coarse_die_plan_v2" ||
              plan_wire.at("schema") == "coarse_die_plan_v3",
          "default Plan keeps its legacy wire version");
  require(!plan_wire.contains("gemm_os_program") &&
              !plan_wire.contains("program") &&
              !legacy_plan->task.contains("runtime_program"),
          "default Plan wire and normalized task omit the opt-in program");
  require(plan_to_json(*plan_from_json(plan_wire)) == plan_wire,
          "legacy Plan wire roundtrips without a program field");

  Runtime runtime(hardware);
  const auto handle = runtime.submit(legacy_plan, owner(50), 0);
  const json snapshot = runtime.export_snapshot();
  require(snapshot.at("schema") == "coarse_die_snapshot_v3" &&
              snapshot.at("snapshot_version") == 3,
          "Runtime with only legacy Plans keeps snapshot v3");
  const json& execution = snapshot.at("executions").at(0);
  require(!execution.contains("program") &&
              !execution.at("plan").contains("program"),
          "legacy execution snapshot does not acquire program state");
  const auto legacy_progress = runtime.query_progress(handle);
  require(!legacy_progress.contains("program") ||
              legacy_progress.at("program").is_null(),
          "legacy execution query does not acquire a program block");
}

void body() {
  compiled_solo_and_arrival_work_conservation();
  ready_launch_and_terminal_wins_same_tick_cancel();
  partial_cancel_preserves_only_completed_cursor_and_served_work();
  capacity_pause_and_resume();
  program_snapshot_prefix_boundaries();
  v4_program_snapshot_import_fork_and_atomic_rejection();
  default_plan_and_snapshot_keep_legacy_wire();
}
}  // namespace

int main() {
  return coarse_die_test::run_test("gemm_os_program_runtime_test", body);
}
