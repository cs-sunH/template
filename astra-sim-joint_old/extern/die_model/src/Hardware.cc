/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
Hardware.cc -- die hardware configuration: the embedded 14-row core table,
strict parsing/validation of the six §3.1 die inputs, resource derivation and
the profile calibration coefficients (design doc §3.1/§6.1/§2.1; contract
§3.1/§3.3).
*/

#include "DieModel.hh"

#include <cmath>
#include <string>
#include <vector>

namespace DieModel {

namespace {

// ---------------------------------------------------------------------------
// Embedded core table (design doc §3.1 table, 14 rows; reference edge lengths
// carry the original micrometer values).  configs/core_table.json mirrors
// these values; configs_consistency_test enforces code == json.

constexpr CoreTableEntry kCoreTable[14] = {
    {0.25, CoreType::Small, 1.9775, 26, 260.3},
    {0.5, CoreType::Small, 3.955, 36, 360.98},
    {0.75, CoreType::Small, 17.7975, 188, 537.42},
    {1.0, CoreType::Small, 69.2125, 740, 702.86},
    {1.25, CoreType::Small, 118.65, 1224, 879.29},
    {1.5, CoreType::Small, 193.795, 2128, 1063.84},
    {2.0, CoreType::Small, 249.165, 2220, 1385.22},
    {0.25, CoreType::Large, 23.73, 204, 388.859567},
    {0.5, CoreType::Large, 23.73, 276, 532.324227},
    {0.75, CoreType::Large, 124.5825, 1218, 787.260242},
    {1.0, CoreType::Large, 249.165, 2700, 1076.754654},
    {1.25, CoreType::Large, 195.7725, 1740, 1142.281725},
    {1.5, CoreType::Large, 348.04, 3798, 1544.891946},
    {2.0, CoreType::Large, 741.5625, 7872, 1759.693467},
};

// ---------------------------------------------------------------------------
// Legal layouts and NoC widths (design doc §3.1): 10 (area, per-side, sides)
// combinations x 5 widths x 14 core rows = 700 candidates.

struct HbmLayout {
  double area_mm2;
  std::uint64_t per_side;
  std::uint64_t sides;
};

constexpr HbmLayout kLegalLayouts[10] = {
    {300, 2, 1}, {300, 2, 2},                      //
    {500, 2, 1}, {500, 3, 1}, {500, 2, 2}, {500, 3, 2},
    {800, 3, 1}, {800, 4, 1}, {800, 3, 2}, {800, 4, 2},
};

constexpr std::uint64_t kLegalNoCWidths[5] = {64, 128, 256, 512, 1024};

// Each HBM stack: 16e9 byte, 410e9 byte/s (design doc §3.1).
constexpr double kHbmCapacityBytes = 16e9;
constexpr double kHbmBandwidthBytes = 410e9;

// Reference NoC bit width per core type (design doc §3.1: small/large
// reference widths 256/512; 15% of the core area scales linearly with
// width).
constexpr double noc_width_ref(CoreType type) {
  return type == CoreType::Small ? 256.0 : 512.0;
}

std::string layout_list() {
  std::string out;
  for (const HbmLayout& l : kLegalLayouts) {
    if (!out.empty()) {
      out += ", ";
    }
    out += "(" + std::to_string(static_cast<int>(l.area_mm2)) + "," +
           std::to_string(l.per_side) + "," + std::to_string(l.sides) + ")";
  }
  return out;
}

}  // namespace

std::vector<CoreTableEntry> core_table() {
  return std::vector<CoreTableEntry>(std::begin(kCoreTable),
                                     std::end(kCoreTable));
}

// ---------------------------------------------------------------------------
// make_hardware

namespace {

Hardware parse_die_node(const nlohmann::json& die, const std::string& path,
                        Profile profile) {
  detail::reject_unknown_keys(
      die,
      {"sram-mb-per-core", "core-type", "compute-area-mm2", "hbm-per-side",
       "hbm-sides", "noc-width-bits"},
      path);

  Hardware hw;
  hw.profile = profile;
  hw.sram_mb_per_core =
      detail::require_number(die, "sram-mb-per-core", path);

  const std::string core_type =
      detail::require_enum_string(die, "core-type", {"small", "large"},
                                  path + ".core-type");
  from_string(core_type, hw.core_type);

  hw.compute_area_mm2 = detail::require_number(die, "compute-area-mm2", path);
  hw.hbm_per_side = detail::require_uint(die, "hbm-per-side", path);
  hw.hbm_sides = detail::require_uint(die, "hbm-sides", path);
  hw.noc_width_bits = detail::require_uint(die, "noc-width-bits", path);

  // Legal (area, per-side, sides) combination (design doc §3.1 list).
  bool layout_ok = false;
  for (const HbmLayout& l : kLegalLayouts) {
    if (hw.compute_area_mm2 == l.area_mm2 && hw.hbm_per_side == l.per_side &&
        hw.hbm_sides == l.sides) {
      layout_ok = true;
      break;
    }
  }
  if (!layout_ok) {
    detail::fail(path,
                 "illegal (compute-area-mm2, hbm-per-side, hbm-sides) = (" +
                     std::to_string(hw.compute_area_mm2) + "," +
                     std::to_string(hw.hbm_per_side) + "," +
                     std::to_string(hw.hbm_sides) +
                     "); legal layouts (design doc §3.1): " + layout_list());
  }

  // NoC width candidate set (design doc §3.1).
  bool width_ok = false;
  for (std::uint64_t w : kLegalNoCWidths) {
    if (hw.noc_width_bits == w) {
      width_ok = true;
      break;
    }
  }
  if (!width_ok) {
    detail::fail(path + ".noc-width-bits",
                 "value " + std::to_string(hw.noc_width_bits) +
                     " not in {64,128,256,512,1024} (design doc §3.1)");
  }

  // (sram, core-type) must hit the core table (design doc §3.1).
  bool row_found = false;
  for (const CoreTableEntry& row : kCoreTable) {
    if (row.sram_mb_per_core == hw.sram_mb_per_core &&
        row.core_type == hw.core_type) {
      hw.core_row = row;
      row_found = true;
      break;
    }
  }
  if (!row_found) {
    detail::fail(path,
                 "no core table row for (sram-mb-per-core, core-type) = (" +
                     std::to_string(hw.sram_mb_per_core) + ", " +
                     to_string(hw.core_type) + ") (design doc §3.1 table)");
  }

  // --- derivation (design doc §3.1 code block; frequency fixed 1 GHz) ---
  const double width = static_cast<double>(hw.noc_width_bits);
  const double width_ref = noc_width_ref(hw.core_type);
  hw.acore_mm2 = (hw.core_row.core_side_ref_um / 1000.0) *
                 (hw.core_row.core_side_ref_um / 1000.0) *
                 (0.85 + 0.15 * width / width_ref);
  const double n_exact = std::sqrt(hw.compute_area_mm2 / hw.acore_mm2);
  hw.mesh_n = static_cast<std::uint64_t>(std::floor(n_exact));
  if (hw.mesh_n == 0) {
    detail::fail(path,
                 "compute-area-mm2 " + std::to_string(hw.compute_area_mm2) +
                     " fits no complete core (Acore " +
                     std::to_string(hw.acore_mm2) + " mm^2, design doc §3.1)");
  }
  hw.cores = hw.mesh_n * hw.mesh_n;
  hw.pe_peak_die =
      static_cast<double>(hw.cores) * hw.core_row.pe_gflops_per_core * 1e9;
  hw.vec_peak_die =
      static_cast<double>(hw.cores) * hw.core_row.vec_g_eqop_per_core * 1e9;
  hw.sram_usable_bytes_per_core = 0.90 * hw.sram_mb_per_core * 1e6;
  hw.link_peak_bytes = width / 8.0 * 1e9;
  hw.hbm_count = hw.hbm_sides * hw.hbm_per_side;
  hw.hbm_capacity_bytes = static_cast<double>(hw.hbm_count) * kHbmCapacityBytes;
  hw.hbm_bandwidth_bytes =
      static_cast<double>(hw.hbm_count) * kHbmBandwidthBytes;

  // --- calibration coefficients (§2.1 formulas via DieTypes.hh; §6.1
  //     defaults for the mesh profile; §2.1 constants for the TX8
  //     reference) ---
  hw.eta_pe = calibration::eta_compute();
  hw.eta_vec = calibration::eta_compute();
  hw.eta_matrix_mem = calibration::kEtaMatrixMem;
  hw.eta_vector_mem = calibration::eta_vector_mem();
  hw.eta_noc = calibration::kEtaNoc;
  hw.launch_us_per_macro_group = calibration::kLaunchUsPerMacroGroup;
  hw.tx8_matrix_peak_flop_s = calibration::matrix_peak_flop_s();
  hw.tx8_vector_peak_eqop_s = calibration::vector_peak_eqop_s();
  hw.tx8_bw_matrix_bytes_s = calibration::bw_matrix_bytes_s();
  hw.tx8_bw_vector_bytes_s = calibration::bw_vector_bytes_s();
  hw.tx8_extra_launch_s = 0.0;              // design doc §2.1
  hw.tx8_extra_memory_latency_s = 0.0;      // design doc §2.1
  hw.full_score_materialization = true;     // design doc §2.1
  hw.skip_fully_masked_tiles = false;       // design doc §2.1
  return hw;
}

}  // namespace

Hardware make_hardware(const nlohmann::json& config) {
  const std::string path = "die";
  if (!config.is_object()) {
    detail::fail(path, "expected a JSON object");
  }

  Profile profile = Profile::MeshDieV1;
  if (config.contains("die")) {
    // Wrapped §3.3-style input.  "derived" is accepted and ignored: it is
    // this library's own output mirror (see hardware_summary_json), so a
    // resolved_die_model.json round-trips through make_hardware.
    detail::reject_unknown_keys(
        config, {"schema_version", "profile", "die", "derived"}, path);
    if (config.contains("schema_version")) {
      const std::uint64_t v =
          detail::require_uint(config, "schema_version", path);
      if (v != 1) {
        detail::fail(path + ".schema_version",
                     "unsupported value " + std::to_string(v) +
                         ", expected 1 (contract §3.3)");
      }
    }
    if (config.contains("profile")) {
      const std::string p = detail::require_enum_string(
          config, "profile", {"mesh_die_v1", "tx8_reference_v1"},
          path + ".profile");
      from_string(p, profile);
    }
    const nlohmann::json& die = detail::member(config, "die", path);
    if (!die.is_object()) {
      detail::fail(path + ".die", "expected a JSON object");
    }
    return parse_die_node(die, path + ".die", profile);
  }

  // Bare die node: exactly the six keys, profile defaults to mesh_die_v1
  // (contract §3.2 default).
  return parse_die_node(config, path, profile);
}

// ---------------------------------------------------------------------------
// hardware_summary_json (contract §3.3)

nlohmann::json hardware_summary_json(const Hardware& hw) {
  nlohmann::json die;
  die["sram-mb-per-core"] = hw.sram_mb_per_core;
  die["core-type"] = to_string(hw.core_type);
  die["compute-area-mm2"] = hw.compute_area_mm2;
  die["hbm-per-side"] = hw.hbm_per_side;
  die["hbm-sides"] = hw.hbm_sides;
  die["noc-width-bits"] = hw.noc_width_bits;

  nlohmann::json derived;
  derived["mesh_n"] = hw.mesh_n;
  derived["cores"] = hw.cores;
  derived["pe_peak_tflops"] = hw.pe_peak_die / 1e12;
  derived["vec_peak_teq"] = hw.vec_peak_die;
  derived["sram_usable_bytes_per_core"] = hw.sram_usable_bytes_per_core;
  derived["link_peak_bytes"] = hw.link_peak_bytes;
  derived["hbm_count"] = hw.hbm_count;
  derived["hbm_capacity_bytes"] = hw.hbm_capacity_bytes;
  derived["hbm_bandwidth_bytes"] = hw.hbm_bandwidth_bytes;

  nlohmann::json out;
  out["schema_version"] = 1;
  out["profile"] = to_string(hw.profile);
  out["die"] = die;
  out["derived"] = derived;
  return out;
}

}  // namespace DieModel
