#include "test_support.hh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <map>
#include <string>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {
using Clock = std::chrono::steady_clock;
constexpr std::uint64_t kNsPerSecond = 1000000000ULL;

struct Args {
  double hbm_bytes_per_s = 1000.0;
  double compute_ops_per_s = 8.0;
  double input_slots = 2.0;
  double output_slots = 2.0;
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
    const std::string value = argv[++i];
    if (key == "--hbm-bytes-per-s") args.hbm_bytes_per_s = std::stod(value);
    else if (key == "--compute-ops-per-s") args.compute_ops_per_s = std::stod(value);
    else if (key == "--input-slots") args.input_slots = std::stod(value);
    else if (key == "--output-slots") args.output_slots = std::stod(value);
    else throw std::runtime_error("unknown argument: " + key);
  }
  if (!std::isfinite(args.hbm_bytes_per_s) || args.hbm_bytes_per_s <= 0.0 ||
      !std::isfinite(args.compute_ops_per_s) || args.compute_ops_per_s <= 0.0 ||
      !std::isfinite(args.input_slots) || args.input_slots < 1.0 ||
      !std::isfinite(args.output_slots) || args.output_slots < 1.0 ||
      std::floor(args.input_slots) != args.input_slots ||
      std::floor(args.output_slots) != args.output_slots)
    throw std::runtime_error("invalid oracle runtime parameters");
  return args;
}

Hardware oracle_hardware(double hbm, double compute) {
  std::array<double, resource_count> rates{};
  rates[COMPUTE_GATE] = compute;
  rates[HBM0] = hbm;
  return manual_hardware(rates);
}

std::shared_ptr<const Plan> grouped_plan(const Hardware& hw,
                                        const json& request,
                                        std::size_t epoch_count,
                                        double input_slots,
                                        double output_slots) {
  const auto& waves = request.at("waves");
  if (!waves.is_array() || epoch_count == 0 || epoch_count > waves.size())
    throw std::runtime_error("invalid grouped-wave plan dimensions");

  auto plan = std::make_shared<Plan>();
  plan->task = json{{"name", request.at("name")},
                    {"arrival_s", request.at("arrival_s")},
                    {"grouped_consumption_waves", epoch_count}};
  plan->summary = json{{"status", "synthetic_oracle_measurement"},
                       {"grouped_consumption_waves", epoch_count}};
  plan->hardware_fingerprint = hw.fingerprint;

  for (std::size_t group = 0; group < epoch_count; ++group) {
    const std::size_t start = group * waves.size() / epoch_count;
    const std::size_t stop = (group + 1) * waves.size() / epoch_count;
    if (stop <= start) throw std::runtime_error("empty grouped-wave epoch");
    double load_bytes = 0.0;
    double compute_ops = 0.0;
    double store_bytes = 0.0;
    for (std::size_t wi = start; wi < stop; ++wi) {
      const auto& wave = waves.at(wi);
      load_bytes += wave.at("load_bytes").get<double>();
      compute_ops += wave.at("compute_ops").get<double>();
      store_bytes += wave.at("store_bytes").get<double>();
    }
    if (std::floor(load_bytes) != load_bytes || std::floor(store_bytes) != store_bytes)
      throw std::runtime_error("fixture HBM byte totals must be integers");

    Epoch epoch;
    epoch.name = "consumption_group_" + std::to_string(group);
    epoch.credit_initial = {{0.0, 0.0}};
    epoch.credit_capacity = {{input_slots, output_slots}};
    epoch.repeat_count = 1;
    const double waves_in_group = static_cast<double>(stop - start);

    Activity load;
    load.id = group * 3 + 1;
    load.role = Role::Load;
    load.work[HBM0] = load_bytes;
    load.hbm_bytes[0][0] = static_cast<std::uint64_t>(load_bytes);
    load.canonical_q = 1.0;
    load.credit_delta[0] = waves_in_group;

    Activity compute;
    compute.id = group * 3 + 2;
    compute.role = Role::Compute;
    compute.work[COMPUTE_GATE] = compute_ops;
    compute.canonical_q = 1.0;
    compute.credit_delta[0] = -waves_in_group;
    compute.credit_delta[1] = waves_in_group;

    Activity store;
    store.id = group * 3 + 3;
    store.role = Role::Store;
    store.work[HBM0] = store_bytes;
    store.hbm_bytes[0][1] = static_cast<std::uint64_t>(store_bytes);
    store.canonical_q = 1.0;
    store.credit_delta[1] = -waves_in_group;

    epoch.activities = {load, compute, store};
    plan->epochs.push_back(std::move(epoch));
  }
  plan->fingerprint = plan_to_json(*plan).at("fingerprint").get<std::string>();
  return plan;
}

json measure(const Hardware& hw, const json& scenario, std::size_t E,
             const Args& args) {
  const auto wall_begin = Clock::now();
  const auto cpu_begin = std::clock();
  Runtime runtime(hw);
  std::map<std::string, std::uint64_t> handles;
  std::map<std::string, std::uint64_t> expected_read, expected_write;

  for (const auto& request : scenario.at("requests")) {
    const std::string name = request.at("name").get<std::string>();
    const auto plan = grouped_plan(hw, request, E, args.input_slots,
                                   args.output_slots);
    const double arrival_s = request.at("arrival_s").get<double>();
    const auto arrival_ns = static_cast<std::uint64_t>(
        std::llround(arrival_s * static_cast<double>(kNsPerSecond)));
    Owner execution_owner;
    const std::uint64_t owner_id = handles.size() + 1;
    execution_owner.request_id = owner_id;
    execution_owner.node_id = owner_id;
    execution_owner.execution_id = owner_id;
    execution_owner.generation = 1;
    handles[name] = runtime.submit(plan, execution_owner, arrival_ns);
    for (const auto& wave : request.at("waves")) {
      expected_read[name] += wave.at("load_bytes").get<std::uint64_t>();
      expected_write[name] += wave.at("store_bytes").get<std::uint64_t>();
    }
  }

  std::size_t advances = 0;
  while (true) {
    bool terminal = true;
    for (const auto& [name, handle] : handles) {
      (void)name;
      if (!runtime.query_progress(handle).at("terminal").get<bool>()) {
        terminal = false;
        break;
      }
    }
    if (terminal) break;
    const auto next = runtime.next_event();
    if (!next || *next <= runtime.now_ns())
      throw std::runtime_error("C++ Runtime oracle measurement has no advancing event");
    runtime.advance_to(*next);
    if (++advances > 1000000)
      throw std::runtime_error("C++ Runtime oracle measurement exceeded event budget");
  }

  json finish = json::object();
  for (const auto& [name, handle] : handles) {
    const auto progress = runtime.query_progress(handle);
    const auto& served = progress.at("served").at("hbm_bytes_by_row_direction");
    require_close(served.at(0).at(0).get<double>(),
                  static_cast<double>(expected_read.at(name)), 1.0e-7,
                  "C++ Runtime must serve every grouped HBM read byte exactly");
    require_close(served.at(0).at(1).get<double>(),
                  static_cast<double>(expected_write.at(name)), 1.0e-7,
                  "C++ Runtime must serve every grouped HBM write byte exactly");
    finish[name] = static_cast<double>(progress.at("finish_ns").get<std::uint64_t>()) /
                   static_cast<double>(kNsPerSecond);
  }
  const auto counters = runtime.counters();
  const auto wall_end = Clock::now();
  const auto cpu_end = std::clock();
  return {{"shape", scenario.at("shape")},
          {"phase_offset_s", scenario.at("phase_offset_s")},
          {"E", E},
          {"finish_by_request_s", finish},
          {"hbm_busy_s", counters.at("hbm_busy_ns").get<double>() /
                              static_cast<double>(kNsPerSecond)},
          {"hbm_shared_s", counters.at("hbm_shared_ns").get<double>() /
                                static_cast<double>(kNsPerSecond)},
          {"internal_events", counters.at("internal_events")},
          {"allocation_updates", counters.at("allocation_updates")},
          {"host_wall_ns", static_cast<std::uint64_t>(
                                std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    wall_end - wall_begin).count())},
          {"host_cpu_ns", static_cast<std::uint64_t>(
                               1.0e9 * static_cast<double>(cpu_end - cpu_begin) /
                               CLOCKS_PER_SEC)}};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    // Test binaries are also used as machine-readable measurement drivers.
    // Parse parameters once, then the test body uses the default scenario set.
    const Args args = parse_args(argc, argv);
    const std::string fixture_path = std::string(COARSE_DIE_SOURCE_DIR) +
                                     "/tests/fixtures/discrete_wave_scenarios.json";
    const json fixture_data = read_json(fixture_path);
    require(fixture_data.at("scenario_count").get<std::size_t>() == 13,
            "shared oracle fixture contains all 13 phase/shape scenarios");
    const auto hw = oracle_hardware(args.hbm_bytes_per_s, args.compute_ops_per_s);
    json measurements = json::array();
    for (const auto& scenario : fixture_data.at("scenarios")) {
      for (const std::size_t E : {5, 8, 16})
        measurements.push_back(measure(hw, scenario, E, args));
    }
    std::cout << json{{"schema_version", 1},
                      {"status", "measured"},
                      {"runtime", "public_coarse_die_cpp_Runtime"},
                      {"hardware_fingerprint", hw.fingerprint},
                      {"hbm_bytes_per_s", args.hbm_bytes_per_s},
                      {"compute_ops_per_s", args.compute_ops_per_s},
                      {"input_slots", args.input_slots},
                      {"output_slots", args.output_slots},
                      {"measurement_count", measurements.size()},
                      {"measurements", std::move(measurements)}}.dump() << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "oracle_granularity_test failed: " << e.what() << "\n";
    return 1;
  }
}
