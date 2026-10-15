#include "test_support.hh"

#include <algorithm>
#include <cmath>
#include <map>
#include <regex>
#include <sstream>
#include <tuple>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {
using Key = std::tuple<int, std::string, double>;
struct SourceRow { int n_vec; int n_pe; };

std::string trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::vector<std::string> cells(const std::string& line) {
  std::vector<std::string> out;
  std::size_t start = 0;
  while (start <= line.size()) {
    const auto end = line.find('|', start);
    if (end == std::string::npos) break;
    if (start != 0) out.push_back(trim(line.substr(start, end - start)));
    start = end + 1;
  }
  return out;
}

std::map<Key, SourceRow> parse_source_table(const std::string& path) {
  std::istringstream input(read_text(path));
  std::string line;
  std::map<Key, SourceRow> rows;
  int active_bits = 0;
  const std::regex bits_re("B[[:space:]]*=[[:space:]]*([0-9]+)[[:space:]]*bit");
  const std::regex first_cell_re("(小|大)[[:space:]]*/[[:space:]]*([0-9]+\\.[0-9]+)");
  while (std::getline(input, line)) {
    if (line.find("NoC 位宽 B =") != std::string::npos) {
      std::smatch match;
      if (!std::regex_search(line, match, bits_re))
        throw std::runtime_error("cannot parse B heading from source markdown");
      active_bits = std::stoi(match[1].str());
      continue;
    }
    if (active_bits == 0 || line.empty() || line.front() != '|') continue;
    const auto columns = cells(line);
    if (columns.size() < 3 || columns[0].find("---") != std::string::npos ||
        columns[0].find("核型") != std::string::npos)
      continue;
    std::smatch match;
    if (!std::regex_search(columns[0], match, first_cell_re)) continue;
    const std::string type = match[1].str() == "小" ? "small" : "large";
    const double sram = std::stod(match[2].str());
    const int n_vec = std::stoi(columns[1]);
    const int n_pe = std::stoi(columns[2]);
    const Key key{active_bits, type, sram};
    if (!rows.emplace(key, SourceRow{n_vec, n_pe}).second)
      throw std::runtime_error("duplicate row in source markdown table");
    if (rows.size() == 64) active_bits = 0;
  }
  return rows;
}

double source_l0(const std::string& core_type, double sram_mib) {
  const double a = sram_mib * 0.40662 / 0.25;
  constexpr double b = 0.13925;
  if (core_type == "small") return a;
  return (a + std::sqrt(a * a - 0.6 * a * b)) / 0.3;
}

json config_for(const json& base, int bits, const std::string& type,
                double sram) {
  json input = base;
  auto& a = input["architecture"];
  a["core_type"] = type;
  a["sram_mib_per_core"] = sram;
  a["noc_width_bits"] = bits;
  const double l0 = source_l0(type, sram);
  a["core_columns_nx"] = 1;
  a["core_rows_ny"] = 1;
  a["hbm_stacks_per_row"] = 2;
  // Keep the documented reticle filters intact. A 1x1 source grid is grown
  // isotropically where the source minimum area requires it; large rows that
  // already exceed the 200 mm^2 requirement use their minimum grid directly.
  a["die_geometry"] = (l0 + 0.1) * (l0 + 0.1) < 200.0
                           ? "isotropic" : "minimum_grid";
  return input;
}

void test_source_table_and_resource_formulas() {
  const auto source_rows = parse_source_table(source_doc("架构建模新.md"));
  require(source_rows.size() == 64,
          "source markdown must provide all 64 N_vec/N_pe rows");
  const auto base = read_json(std::string(COARSE_DIE_SOURCE_DIR) +
                              "/configs/example_hardware.json");
  const auto core64 = read_json(std::string(COARSE_DIE_SOURCE_DIR) +
                                "/configs/core64.json");
  require(core64.at("schema") == "coarse_die_core64_v2" &&
              core64.at("row_count").get<std::size_t>() == 64 &&
              core64.at("rows").size() == 64,
          "core64.json must retain all rows as an independently reviewable fixture");
  const std::string source_digest =
      "8e7d393ff9a96dd556088fa7c98a6fd80ed0b277e03cde7c3909a0f118e45f36";
  require(core64.at("rows_sha256") == source_digest &&
              core64.at("table_identity").at("rows_sha256") == source_digest,
          "core64.json must match the frozen serialized row digest");
  for (const auto& row : core64.at("rows")) {
    const Key key{row.at("noc_width_bits").get<int>(),
                  row.at("core_type").get<std::string>(),
                  row.at("sram_mib_per_core").get<double>()};
    const auto source = source_rows.find(key);
    require(source != source_rows.end(), "core64.json has a row absent from source markdown");
    require(row.at("n_vec").get<int>() == source->second.n_vec &&
                row.at("n_pe").get<int>() == source->second.n_pe,
            "core64.json N_vec/N_pe differs from original markdown");
    require_close(row.at("l0_mm_display").get<double>(),
                  std::round(source_l0(std::get<1>(key), std::get<2>(key)) * 1.0e5) / 1.0e5,
                  1.0e-12, "core64.json display edge differs from source formula");
  }
  require(base.at("core_resource_table").at("id") == "core64_v2",
          "fixture must identify the immutable source resource table");
  require(base.at("core_resource_table").at("rows_sha256") == source_digest,
          "fixture row digest must match the independently frozen source golden");
  const std::array<int, 4> widths{{64, 128, 256, 512}};
  const std::array<double, 8> capacities{{0.25, 0.50, 0.75, 1.00,
                                           1.25, 1.50, 1.75, 2.00}};
  for (const int bits : widths) {
    for (const std::string type : {"small", "large"}) {
      for (const double sram : capacities) {
        const auto row_it = source_rows.find(Key{bits, type, sram});
        require(row_it != source_rows.end(), "source table key missing");
        const auto hw = resolve_hardware(config_for(base, bits, type, sram));
        const auto& core = hw.derived.at("core");
        require(core.at("N_vec").get<int>() == row_it->second.n_vec,
                "compiled N_vec differs from original source markdown");
        require(core.at("N_pe").get<int>() == row_it->second.n_pe,
                "compiled N_pe differs from original source markdown");
        require(core.at("N_pe").get<int>() == static_cast<int>(
                    std::floor(19.777 * row_it->second.n_vec + 0.5)),
                "source N_pe must use round-half-up 19.777*N_vec");
        require_close(core.at("L0_mm").get<double>(), source_l0(type, sram),
                      1.0e-10, "Core edge must be calculated from source geometry");
        require(core.at("P_int4_per_core_ops_per_s").get<double>() ==
                    2.0 * row_it->second.n_pe * 1.0e9,
                "INT4 PE peak must stay separate and exact");
        require_close(core.at("P_vec_per_core_ops_per_s").get<double>(),
                      3.9554 * row_it->second.n_vec * 1.0e9, 1.0e-3,
                      "Vec FP32-equivalent peak must use source lane rate");
        const double expectedLocal =
            core.at("N_core").get<double>() * (bits / 8.0) * 1.0e9 * 0.70;
        require_close(hw.rates[LOCAL], expectedLocal, 1.0e-3,
                      "B must affect the local-bus service rate under the stated profile");
        require_close(core.at("L0_mm").get<double>(), source_l0(type, sram),
                      1.0e-10,
                      "changing B must not recompute the fixed baseline Core edge");
      }
    }
  }
  const auto baseline = resolve_hardware(base);
  auto bad_hash = base;
  bad_hash["core_resource_table"]["rows_sha256"] = "wrong";
  bool hash_rejected = false;
  try { (void)resolve_hardware(bad_hash); }
  catch (const std::invalid_argument&) { hash_rejected = true; }
  require(hash_rejected, "resolver must reject a mismatched Core64 config golden");
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
              system.at("module_rows").get<int>() == 4,
          "documented 215x215 mm region must fit a 9x4 module grid");
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
  std::cout << "source markdown rows=64, source hash="
            << baseline.derived.at("core_resource_table").at("rows_sha256")
            << ", baseline cores=88, HBM stacks=4, capacity=144000000000 B, "
               "peak=4710400000000 B/s\n";
}
}  // namespace

int main() { return run_test("source_golden_test", test_source_table_and_resource_formulas); }
