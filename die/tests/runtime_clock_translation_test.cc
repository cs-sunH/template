#include "test_support.hh"
#include "../src/GemmOSResourceFacts.hh"

#include <array>
#include <limits>

using namespace coarse_die;
using namespace coarse_die_test;
namespace access = coarse_die::detail::gemm_os_access_counter;

static Hardware clock_hardware() {
  auto config = read_json(std::string(COARSE_DIE_SOURCE_DIR) +
                          "/configs/example_hardware.json");
  auto& a = config["architecture"];
  a["sram_mib_per_core"] = 0.25;
  a["noc_width_bits"] = 64; a["core_columns_nx"] = 1;
  a["core_rows_ny"] = 1; a["die_geometry"] = "isotropic";
  config["performance_profile"]["eta_vec"] = 0.02;
  config["performance_profile"]["eta_hbm_matrix"] = 1.0;
  auto hw = resolve_hardware(config);
  for (std::size_t r = 0; r < resource_count; ++r) {
    if (!hw.derived["resources"][r]["enabled"].get<bool>()) continue;
    hw.rates[r] = r == COMPUTE_GATE ? 1.0 :
        (r == HBM0 || r == HBM1 ? 50000000.0 : 1e30);
    hw.derived["resources"][r]["rate"] = hw.rates[r];
  }
  hw.fingerprint = "clock-translation-synthetic-single-core-v1";
  return hw;
}

static std::shared_ptr<const Plan> stage(const Hardware& hw, Activity fact) {
  auto p = std::make_shared<Plan>();
  p->task = json::object(); p->summary = json::object();
  p->hardware_fingerprint = hw.fingerprint;
  Epoch e; e.name = "single_nonfinal_stage"; e.activities = {std::move(fact)};
  p->epochs = {std::move(e)};
  p->fingerprint = plan_to_json(*p).at("fingerprint");
  return p;
}

static json exercise(const Hardware& hw,
                     const std::array<std::shared_ptr<const Plan>,2>& plans,
                     std::uint64_t start, unsigned mode) {
  Runtime runtime(hw); runtime.advance_to(start);
  std::uint64_t elapsed = 0;
  json terminals = json::array();
  const unsigned cycles = mode == 0 ? 32 : 1;
  Owner owner; owner.node_id = 1; owner.execution_id = 1;
  for (unsigned i = 0; i < cycles; ++i) {
    const auto begin = runtime.now_ns();
    const auto load = runtime.submit(plans[0], owner, begin);
    const auto compute = runtime.submit(plans[1], owner, begin);
    if (mode != 0) {
      runtime.advance_to(begin + 123);
      auto imported = Runtime::import_snapshot(hw, runtime.export_snapshot());
      require(imported->export_snapshot() == runtime.export_snapshot(),
              "partial snapshot roundtrip preserves old v3 wire");
      if (mode == 1) {
        runtime.cancel(load, begin + 123); imported->cancel(load, begin + 123);
      } else {
        runtime.update_capacity(HBM0, hw.rates[HBM0]/2, begin + 123);
        imported->update_capacity(HBM0, hw.rates[HBM0]/2, begin + 123);
      }
      while (const auto next = runtime.next_event()) {
        runtime.advance_to(*next); imported->advance_to(*next);
        require(imported->export_snapshot() == runtime.export_snapshot(),
                "source/import continuation is identical after clock translation");
      }
    } else {
      unsigned guard = 0;
      while (!runtime.terminal_finish_ns(load) || !runtime.terminal_finish_ns(compute)) {
        const auto next = runtime.next_event();
        require(next.has_value() && ++guard <= 4, "bounded two-stage witness events");
        runtime.advance_to(*next);
      }
    }
    terminals.push_back({*runtime.terminal_finish_ns(load)-begin,
                         *runtime.terminal_finish_ns(compute)-begin});
    elapsed = runtime.now_ns() - start;
    runtime.destroy_execution(load); runtime.destroy_execution(compute);
  }
  auto counters = runtime.counters(); counters.erase("now_ns");
  auto stats = runtime.export_snapshot().at("stats");
  return {{"elapsed_ns",elapsed},{"relative_terminals",terminals},
          {"counters",counters},{"stats_exact_strings",stats}};
}

int main() {
  const auto hw = clock_hardware();
  const std::array<std::uint64_t,6> starts{{0,50000000ULL,1000000000000ULL,
      9007199254740991ULL,9223372036854775808ULL,
      std::numeric_limits<std::uint64_t>::max()-30000000ULL}};
  std::size_t checked = 0;
  for (auto mn : {std::array<std::uint64_t,2>{16,64},{16,1},{1,64},{1,1}}) {
    access::Input in; in.m = mn[0]; in.n = mn[1]; in.k = 64;
    in.matrix_path = access::MatrixPath::StrictFp32Vec;
    const auto facts = detail::gemm_os_single_core_resource_facts(hw,in);
    const std::array<std::shared_ptr<const Plan>,2> plans{{stage(hw,facts[0]),stage(hw,facts[1])}};
    for (unsigned mode = 0; mode < 3; ++mode) {
      const auto reference = exercise(hw,plans,0,mode);
      for (const auto start : starts) {
        require(exercise(hw,plans,start,mode) == reference,
                "relative completion/progress/busy/served must not depend on absolute clock");
        ++checked;
      }
    }
  }
  // Existing v3 supports fractional legacy imports. Preserve their declared
  // scalar precision and settle the remaining fraction at the public tick.
  Runtime empty(hw); auto fractional = empty.export_snapshot();
  fractional["now_ns"] = 1; fractional["exact_ns"] = "0.5";
  auto legacy = Runtime::import_snapshot(hw,fractional);
  require(std::stold(legacy->export_snapshot().at("exact_ns").get<std::string>()) == 0.5L,
          "legacy fractional clock roundtrip");
  legacy->advance_to(1);
  require(std::stold(legacy->export_snapshot().at("exact_ns").get<std::string>()) == 1.0L,
          "fractional legacy tick closes exactly");
  fractional["now_ns"] = 1099511627777ULL;
  fractional["exact_ns"] = "1099511627776.5";
  auto large_legacy = Runtime::import_snapshot(hw,fractional);
  require(std::stold(large_legacy->export_snapshot().at("exact_ns").get<std::string>()) ==
              1099511627776.5L, "large-origin representable legacy fraction roundtrip");
  large_legacy->advance_to(1099511627793ULL);
  require(large_legacy->now_ns() == 1099511627793ULL,
          "large-origin fractional legacy continuation");

  access::Input in; in.m=16; in.n=64; in.k=64;
  in.matrix_path=access::MatrixPath::StrictFp32Vec;
  const auto load_plan=stage(hw,detail::gemm_os_single_core_resource_facts(hw,in)[0]);
  Runtime edge(hw); const auto maximum=std::numeric_limits<std::uint64_t>::max();
  edge.advance_to(maximum-10);
  const auto handle=edge.submit(load_plan,Owner{},maximum-10);
  const auto before=edge.export_snapshot(); bool rejected=false;
  try { (void)edge.next_event(); } catch (const std::overflow_error&) { rejected=true; }
  require(rejected && edge.export_snapshot()==before,"next-event u64 overflow is read-only");
  edge.advance_to(maximum); edge.cancel(handle,maximum);
  require(edge.terminal_finish_ns(handle)==std::optional<std::uint64_t>(maximum),
          "partial work can cancel at the final uint64 tick");
  std::cout << "PASS: " << checked << " translated real-fact runs, partial import/cancel/capacity, legacy fractional wire and u64 edge\n";
}
