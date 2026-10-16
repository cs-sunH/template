#include "test_support.hh"

#include <algorithm>
#include <cmath>
#include <set>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {
double source_l0(double sram_mib) {
  return sram_mib * 0.40662 / 0.25;
}

json config_for(const json& base, int bits, double sram) {
  json input = base;
  auto& a = input["architecture"];
  a["sram_mib_per_core"] = sram;
  a["noc_width_bits"] = bits;
  a["core_columns_nx"] = 1;
  a["core_rows_ny"] = 1;
  return input;
}

void test_source_table_and_resource_formulas() {
  const auto base = read_json(std::string(COARSE_DIE_SOURCE_DIR) +
                              "/configs/example_hardware.json");
  const auto table = read_json(std::string(COARSE_DIE_SOURCE_DIR) +
                               "/configs/core32.json");
  require(table.at("schema") == "coarse_die_core32_v1" &&
              table.at("row_count") == 32 && table.at("rows").size() == 32,
          "Core32 contains exactly four NoC widths and eight SRAM capacities");
  const std::string source_digest =
      "d66b4a76122742ed91102d75305e01f10a8241213c96ed2289ecd51a1f7aa346";
  require(table.at("rows_sha256") == source_digest &&
              table.at("table_identity").at("rows_sha256") == source_digest,
          "resource rows must match the frozen DOCX table identity");
  require(base.at("core_resource_table").at("id") == "core32_v1" &&
              base.at("core_resource_table").at("rows_sha256") == source_digest,
          "fixture must identify the immutable source table");
  std::set<std::pair<int, double>> seen;
  for (const auto& row : table.at("rows")) {
    const int bits = row.at("noc_width_bits").get<int>();
    const double sram = row.at("sram_mib_per_core").get<double>();
    require(seen.emplace(bits, sram).second, "source table keys must be unique");
    require(!row.contains("core_type"), "resource rows have no Core type selector");
    const auto hw = resolve_hardware(config_for(base, bits, sram));
    const auto& core = hw.derived.at("core");
    require(!core.contains("core_type"), "derived Core has no type selector");
    require(core.at("N_vec") == row.at("n_vec") &&
                core.at("N_pe") == row.at("n_pe"),
            "compiled resources must match all DOCX-derived rows");
    require(core.at("N_pe").get<int>() == static_cast<int>(
                std::floor(19.777 * row.at("n_vec").get<int>() + 0.5)),
            "PE count must use source round-half-up rule");
    require_close(core.at("L0_mm").get<double>(), source_l0(sram),
                  1.0e-10, "Core edge equals the SRAM physical long side");
    require_close(row.at("l0_mm_display").get<double>(), source_l0(sram),
                  1.0e-10, "display edge must match source geometry");
    require(core.at("P_int4_per_core_ops_per_s").get<double>() ==
                2.0 * row.at("n_pe").get<int>() * 1.0e9,
            "INT4 PE peak stays exact and separate");
    require_close(core.at("P_vec_per_core_ops_per_s").get<double>(),
                  3.9554 * row.at("n_vec").get<int>() * 1.0e9, 1.0e-3,
                  "Vec peak uses the source FP32 equivalent lane rate");
    require_close(hw.rates[LOCAL], (bits / 8.0) * 1.0e9 * 0.70, 1.0e-3,
                  "NoC width affects local bandwidth for the fixed Core");
  }
  const auto baseline = resolve_hardware(base);
  auto bad_hash = base;
  bad_hash["core_resource_table"]["rows_sha256"] = "wrong";
  bool hash_rejected = false;
  try { (void)resolve_hardware(bad_hash); }
  catch (const std::invalid_argument&) { hash_rejected = true; }
  require(hash_rejected, "resolver must reject a mismatched Core32 config golden");
  auto bad_constant = base;
  bad_constant["source_constants"]["hbm_stack_capacity_bytes"] = 35.0e9;
  bool constant_rejected = false;
  try { (void)resolve_hardware(bad_constant); }
  catch (const std::invalid_argument&) { constant_rejected = true; }
  require(constant_rejected, "resolver must reject changed source constants");
  require(baseline.derived.at("core").at("N_core").get<int>() == 88,
          "documented 8x11 rectangular grid must have 88 cores");
  require(baseline.derived.at("hbm").at("total_stacks").get<int>() == 4,
          "two stacks per row must yield four HBM stacks");
  require(baseline.derived.at("hbm").at("capacity_bytes").get<double>() ==
              144.0e9,
          "four HBM stacks must provide 144 GB in decimal bytes");
  require(baseline.derived.at("hbm").at("peak_Bps").get<double>() ==
              4710.4e9,
          "four HBM stacks must provide 4710.4 GB/s raw peak");
  const auto& die = baseline.derived.at("die");
  const double l0 = baseline.derived.at("core").at("L0_mm").get<double>();
  require_close(die.at("minimum_width_mm").get<double>(), 8 * l0 + 0.10,
                1.0e-10, "x-axis must reserve 0.10 mm total peripheral width");
  require_close(die.at("minimum_height_mm").get<double>(), 11 * l0 + 0.10,
                1.0e-10, "y-axis must reserve 0.10 mm total peripheral height");
  const double bump = die.at("area_mm2").get<double>() /
                      (0.055 * 0.055) * 0.40 * 2.5e9 / 8.0;
  require_close(die.at("bump_budget_Bps").get<double>(), bump, 1.0e-3,
                "bump budget must use the original area-density formula");
  require_close(die.at("d2d_bidirectional_Bps").get<double>(),
                bump - baseline.derived.at("hbm").at("peak_Bps").get<double>(),
                1.0e-3, "D2D budget must subtract theoretical HBM capacity");
  const double hbmRowWidth = 2.0 * 10.98 + 0.15;
  require_close(baseline.derived.at("hbm").at("row_width_mm").get<double>(),
                hbmRowWidth, 1.0e-12,
                "two-stack row geometry must use the source HBM gap");
  require_close(baseline.derived.at("hbm").at("module_width_mm").get<double>(),
                std::max(die.at("width_mm").get<double>(), hbmRowWidth) + 0.30,
                1.0e-12, "module outline must use max(Core Die, HBM row) plus margins");
  const auto& system = baseline.derived.at("system");
  require(system.at("module_columns").get<int>() == 9 &&
              system.at("module_rows").get<int>() == 3,
          "the explicit 21x29 mm example fits a 9x3 module grid");
  require_close(system.at("effective_module_width_mm").get<double>(),
                baseline.derived.at("hbm").at("module_width_mm").get<double>() /
                    std::sqrt(0.90),
                1.0e-12, "10% module interconnect must expand the system footprint once");

  const auto same_side_json = [&] {
    auto input = base;
    input["architecture"]["hbm_placement_mode"] = "same_side_two_rows";
    return input;
  }();
  const auto same_side = resolve_hardware(same_side_json);
  require(same_side.derived.at("die") == baseline.derived.at("die"),
          "HBM placement must not alter die geometry or bump budget");
  require(same_side.derived.at("hbm").at("capacity_bytes") ==
              baseline.derived.at("hbm").at("capacity_bytes"),
          "placement must not alter HBM capacity");
  require(same_side.derived.at("hbm").at("peak_Bps") ==
              baseline.derived.at("hbm").at("peak_Bps"),
          "placement must not alter HBM raw peak");

  auto supplied_input = base;
  supplied_input["architecture"]["die_geometry"] = "explicit";
  supplied_input["architecture"]["die_width_mm"] = baseline.derived.at("die").at("width_mm");
  supplied_input["architecture"]["die_height_mm"] = baseline.derived.at("die").at("height_mm");
  supplied_input.erase("search_profile");
  const auto supplied = resolve_hardware(supplied_input);
  require(supplied.rates == baseline.rates &&
              supplied.derived.at("core") == baseline.derived.at("core"),
          "explicit user hardware preserves source resource formulas without a search");

  for (const int n : {1, 2, 3}) {
    auto input = base;
    input["architecture"]["hbm_stacks_per_row"] = n;
    const auto hw = resolve_hardware(input);
    const auto& hbm = hw.derived.at("hbm");
    require(hbm.at("total_stacks").get<int>() == 2 * n,
            "n must produce 2n HBM stacks");
    require(hbm.at("capacity_bytes").get<double>() == 2.0 * n * 36.0e9,
            "HBM capacity must scale as 2n*36e9 bytes");
    require(hbm.at("peak_Bps").get<double>() == 2.0 * n * 1.1776e12,
            "HBM raw peak must scale as 2n*1177.6 GB/s");
  }
  std::cout << "source DOCX rows=32, source hash="
            << baseline.derived.at("core_resource_table").at("rows_sha256")
            << ", baseline cores=88, HBM stacks=4, capacity=144000000000 B, "
               "peak=4710400000000 B/s\n";
}
}  // namespace

int main() { return run_test("source_golden_test", test_source_table_and_resource_formulas); }
