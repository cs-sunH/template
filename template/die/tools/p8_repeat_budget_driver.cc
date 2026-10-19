// P8 large-R repeat budget measurement driver (research track, 2026-10-17).
//
// Measures two paths over declared serial_on_drain repeat plans:
//   (a) runtime advance: one submitted compiled plan drained through every
//       declared round via the public next_event/advance_to loop;
//   (b) compile ranking: the per-candidate trial stream that compile() runs
//       internally, reconstructed exactly through normalize_task +
//       enumerate_mappings_for_compile + one public Runtime drain per
//       candidate (the same submit + advance events the ranking executes).
//
// Event counts are exact integers read from Runtime::counters()
// ("internal_events"); wall times use steady_clock. The caller interleaves
// the R levels across rounds (alternating sweep direction per round) so any
// monotone machine drift hits every level equally.
//
// Build (orchestrated by tools/run_p8_budget.py, strict-matrix precedent):
//   g++ -std=c++17 -O3 -DNDEBUG -I die/src -I die/include \
//       -I astra-sim-wscllm/extern/helper <this file> <libcoarse_die.a>
#include "CompilerInternal.hh"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "coarse_die/Execution.hh"
#include "coarse_die/Hardware.hh"
#include "coarse_die/Plan.hh"

namespace coarse_die::detail {
// Defined in Compiler.cc; re-declared so this driver enumerates the exact
// candidate set the compile ranking trials run over. Test-driver precedent:
// tests include ../src/CompilerInternal.hh for normalize_task; this adds the
// one ranking entry point the compiler keeps file-local.
std::vector<std::shared_ptr<Plan>> enumerate_mappings_for_compile(
    const Hardware& hardware, const nlohmann::json& normalized_task);
}

namespace {

using namespace coarse_die;
using nlohmann::json;

json read_json(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  json value;
  in >> value;
  return value;
}

struct DrainResult {
  std::uint64_t events = 0;
  std::uint64_t iterations = 0;
  std::uint64_t finish_ns = 0;
  std::int64_t wall_ns = 0;
};

DrainResult drain(std::shared_ptr<const Hardware> hardware,
                  std::shared_ptr<const Plan> plan) {
  const auto start = std::chrono::steady_clock::now();
  Runtime runtime(std::move(hardware));
  const auto handle = runtime.submit(std::move(plan), Owner{1, 1, 1, 1}, 0);
  std::uint64_t iterations = 0;
  while (!runtime.drained()) {
    const auto next = runtime.next_event();
    if (!next || *next <= runtime.now_ns())
      throw std::runtime_error("p8 driver: drain stalled without a next event");
    runtime.advance_to(*next);
    if (++iterations > 4000000)
      throw std::runtime_error("p8 driver: drain exceeded iteration guard");
  }
  const auto finish = runtime.terminal_finish_ns(handle);
  const auto wall = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - start).count();
  DrainResult result;
  result.events = runtime.counters().at("internal_events").get<std::uint64_t>();
  result.iterations = iterations;
  result.finish_ns = finish.value_or(0);
  result.wall_ns = wall;
  return result;
}

json measure_level(const Hardware& hardware,
                   std::shared_ptr<const Hardware> shared,
                   std::uint64_t layers, bool with_reconstruction) {
  json task{{"op_kind", "ffn_region"},
            {"tensor_shapes", {{"m", 1}, {"hidden", 8}, {"ffn", 16},
                               {"mlp_variant", "swiglu"}}},
            {"fusion_policy", "materialized"},
            {"layer_instances", layers},
            {"weight_passes", std::uint64_t{1}},
            {"repeat_seams", {{"mode", "serial_on_drain"}}}};
  json row;
  row["level_layers"] = layers;

  const auto compile_start = std::chrono::steady_clock::now();
  const auto plan = compile(hardware, task);
  row["compile_wall_ns"] = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - compile_start).count();
  row["candidate_count"] = plan->summary.at("candidate_count");
  row["winner_latency_ns"] = plan->summary.at("estimated_latency_ns");
  row["repeat_count"] = plan->repeat ? json(plan->repeat->count) : json(nullptr);
  row["seam_ns"] =
      plan->repeat ? json(plan->repeat->seam_delay_ns) : json(nullptr);
  row["launch_ns"] = plan->launch_ns;

  // (a) runtime advance path on the compiled plan.
  const auto advance = drain(shared, plan);
  row["advance_events"] = advance.events;
  row["advance_iterations"] = advance.iterations;
  row["advance_finish_ns"] = advance.finish_ns;
  row["advance_wall_ns"] = advance.wall_ns;

  if (with_reconstruction) {
    const auto normalized = detail::normalize_task(hardware, task);
    const auto candidates =
        detail::enumerate_mappings_for_compile(hardware, normalized);
    std::uint64_t ranking_events = 0;
    std::int64_t trials_wall_ns = 0;
    json per_candidate = json::array();
    for (const auto& candidate : candidates) {
      const auto trial = drain(shared, candidate);
      ranking_events += trial.events;
      trials_wall_ns += trial.wall_ns;
      per_candidate.push_back({{"latency_ns", trial.finish_ns},
                               {"events", trial.events},
                               {"iterations", trial.iterations},
                               {"wall_ns", trial.wall_ns},
                               {"exceeds_estimate_event_budget",
                                trial.events > 1000000}});
    }
    row["reconstructed_candidates"] = candidates.size();
    row["ranking_events_total"] = ranking_events;
    row["ranking_trials_wall_ns"] = trials_wall_ns;
    row["per_candidate"] = per_candidate;
    std::uint64_t best = ~std::uint64_t{0};
    for (const auto& entry : per_candidate)
      best = std::min(best, entry.at("latency_ns").get<std::uint64_t>());
    row["cross_check_ok"] =
        (candidates.size() ==
         plan->summary.at("candidate_count").get<std::uint64_t>()) &&
        best == plan->summary.at("estimated_latency_ns").get<std::uint64_t>() &&
        advance.finish_ns ==
            plan->summary.at("estimated_latency_ns").get<std::uint64_t>();
  }
  return row;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: p8_repeat_budget_driver <hardware.json> <rounds> "
                 "<reconstruct:0|1> <R> [<R> ...]\n";
    return 2;
  }
  try {
    const Hardware hardware = resolve_hardware(read_json(argv[1]));
    const auto shared = std::make_shared<const Hardware>(hardware);
    const int rounds = std::stoi(argv[2]);
    const bool reconstruct = std::string(argv[3]) == "1";
    std::vector<std::uint64_t> levels;
    for (int i = 4; i < argc; ++i) levels.push_back(std::stoull(argv[i]));

    json out;
    out["hardware_fingerprint"] = hardware.fingerprint;
    out["rounds"] = rounds;
    out["levels"] = levels;
    json rows = json::array();
    for (int round = 0; round < rounds; ++round) {
      // Alternate the sweep direction every round so any monotone machine
      // drift hits every level equally (same-window interleaved discipline).
      for (std::size_t index = 0; index < levels.size(); ++index) {
        const std::size_t slot =
            round % 2 == 0 ? index : levels.size() - 1 - index;
        rows.push_back(measure_level(hardware, shared, levels[slot],
                                     reconstruct && round == 0));
      }
    }
    out["rows"] = rows;
    std::cout << out.dump(2) << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "p8 driver error: " << error.what() << "\n";
    return 1;
  }
}
