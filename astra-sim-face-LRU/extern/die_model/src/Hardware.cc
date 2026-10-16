/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
Hardware.cc -- die hardware configuration: the embedded 32-row core table,
strict parsing/validation of the explicit die inputs, resource derivation and
legacy scalar/TX8 profile constants. Core resources follow the current
DOCX §1.2/§1.3; legacy area/HBM restrictions are compatibility inputs only.
*/

#include "DieModel.hh"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace DieModel {

namespace {

// ---------------------------------------------------------------------------
// Embedded Core table (current DOCX §1.3, 32 rows; SRAM long-side lengths
// carry the original micrometer values).  configs/core_table.json mirrors
// these values; configs_consistency_test enforces code == json.

constexpr CoreTableEntry kCoreTable[32] = {
    {0.25, 64, 7.9108, 80.0, 406.62},
    {0.5, 64, 59.331, 594.0, 813.24},
    {0.75, 64, 158.216, 1582.0, 1219.86},
    {1.0, 64, 304.5658, 3046.0, 1626.48},
    {1.25, 64, 498.3804, 4984.0, 2033.1000000000001},
    {1.5, 64, 806.9016, 8070.0, 2439.72},
    {1.75, 64, 1107.512, 11076.0, 2846.34},
    {2.0, 64, 1455.5872, 14556.0, 3252.96},
    {0.25, 128, 7.9108, 80.0, 406.62},
    {0.5, 128, 59.331, 594.0, 813.24},
    {0.75, 128, 158.216, 1582.0, 1219.86},
    {1.0, 128, 304.5658, 3046.0, 1626.48},
    {1.25, 128, 498.3804, 4984.0, 2033.1000000000001},
    {1.5, 128, 783.1692, 7832.0, 2439.72},
    {1.75, 128, 1079.8242, 10798.0, 2846.34},
    {2.0, 128, 1423.944, 14240.0, 3252.96},
    {0.25, 256, 7.9108, 80.0, 406.62},
    {0.5, 256, 59.331, 594.0, 813.24},
    {0.75, 256, 158.216, 1582.0, 1219.86},
    {1.0, 256, 276.878, 2768.0, 1626.48},
    {1.25, 256, 462.7818, 4628.0, 2033.1000000000001},
    {1.5, 256, 711.972, 7120.0, 2439.72},
    {1.75, 256, 996.7608, 9968.0, 2846.34},
    {2.0, 256, 1329.0144, 13290.0, 3252.96},
    {0.25, 512, 7.9108, 80.0, 406.62},
    {0.5, 512, 47.4648, 474.0, 813.24},
    {0.75, 512, 126.5728, 1266.0, 1219.86},
    {1.0, 512, 221.5024, 2216.0, 1626.48},
    {1.25, 512, 391.5846, 3916.0, 2033.1000000000001},
    {1.5, 512, 617.0424, 6170.0, 2439.72},
    {1.75, 512, 830.634, 8306.0, 2846.34},
    {2.0, 512, 1075.8688, 10758.0, 3252.96},
};

// ---------------------------------------------------------------------------
// Historical mesh_die_v1 area/HBM layout restrictions are retained only
// for the legacy scalar compatibility profile, not current DOCX hardware
// requirements. Four NoC widths select current single-design Core rows.

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

constexpr std::uint64_t kLegalNoCWidths[4] = {64, 128, 256, 512};

// Explicit historical mesh_die_v1 HBM reference per stack: 16e9 B and
// 410e9 B/s. This is not the current DOCX HBM3E source or a target measurement.
constexpr double kHbmCapacityBytes = 16e9;
constexpr double kHbmBandwidthBytes = 410e9;

// The authoritative Core outline is fixed by the SRAM long side.
// NoC width selects installed PE/Vec resources, without resizing the Core.

nlohmann::json source_identity_json() {
  static const auto identity = nlohmann::json::parse(R"source_identity({"source_revision":"41f119135fd2d4cc87292f59b4b00adc918eece0","source_document":{"path":"架构建模_单Core版_严格复核修订.docx","sha256":"b4382bc7a655b7a3bf9a531b583ce8b52f8d1036883796db8a52b26b4cae4ad3"},"core_resource_table":{"id":"core32_v1","rows_sha256":"d66b4a76122742ed91102d75305e01f10a8241213c96ed2289ecd51a1f7aa346"},"model_scope":"legacy_scalar_compatibility; INT4 architectural peak and FP32-equivalent Vec; target hardware not calibrated"})source_identity");
  return identity;
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
      {"sram-mb-per-core", "mesh-nx", "mesh-ny", "compute-area-mm2", "die-width-mm", "die-height-mm", "hbm-per-side",
       "hbm-sides", "noc-width-bits"},
      path);

  Hardware hw;
  hw.profile = profile;
  hw.sram_mb_per_core =
      detail::require_number(die, "sram-mb-per-core", path);


  if (!die.contains("mesh-nx") || !die.contains("mesh-ny")) {
    detail::fail(path, "explicit mesh-nx and mesh-ny are required; hardware grids are never inferred from area; use coarse_die_v2 for rectangular grids");
  }
  const auto nx = detail::require_uint(die, "mesh-nx", path);
  const auto ny = detail::require_uint(die, "mesh-ny", path);
  if (nx == 0 || ny == 0 || nx != ny) {
    detail::fail(path, "mesh-nx and mesh-ny must be equal positive integers in the legacy scalar backend; use coarse_die_v2 for rectangular grids");
  }
  hw.mesh_n = nx;
  hw.compute_area_mm2 = detail::require_number(die, "compute-area-mm2", path);
  const bool has_width = die.contains("die-width-mm");
  const bool has_height = die.contains("die-height-mm");
  if (has_width != has_height) {
    detail::fail(path, "die-width-mm and die-height-mm must be provided together");
  }
  if (has_width) {
    hw.explicit_die_dimensions = true;
    hw.die_width_mm = detail::require_number(die, "die-width-mm", path);
    hw.die_height_mm = detail::require_number(die, "die-height-mm", path);
    if (!(hw.die_width_mm > 0.0) || !(hw.die_height_mm > 0.0)) {
      detail::fail(path, "die-width-mm and die-height-mm must be positive");
    }
    const long double physical_area = static_cast<long double>(hw.die_width_mm) * hw.die_height_mm;
    const long double difference = std::abs(physical_area - hw.compute_area_mm2);
    if (!std::isfinite(physical_area) ||
        difference > 1e-12L * std::max(1.0L, std::abs(static_cast<long double>(hw.compute_area_mm2)))) {
      detail::fail(path, "die-width-mm times die-height-mm must equal compute-area-mm2");
    }
  }
  hw.hbm_per_side = detail::require_uint(die, "hbm-per-side", path);
  hw.hbm_sides = detail::require_uint(die, "hbm-sides", path);
  hw.noc_width_bits = detail::require_uint(die, "noc-width-bits", path);

  // Legacy scalar profile area/HBM compatibility restrictions.
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
                     "); legal layouts (legacy mesh_die_v1 compatibility profile): " + layout_list());
  }

  // Current DOCX Core table supports four fixed NoC width values.
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
                     " not in {64,128,256,512} (current DOCX Core table)");
  }

  // Match the unique Core resource row by SRAM capacity and NoC width.
  bool row_found = false;
  for (const CoreTableEntry& row : kCoreTable) {
    if (row.sram_mb_per_core == hw.sram_mb_per_core &&
        row.noc_width_bits == hw.noc_width_bits) {
      hw.core_row = row;
      row_found = true;
      break;
    }
  }
  if (!row_found) {
    detail::fail(path,
                 "no core table row for (sram-mb-per-core, noc-width-bits) = (" +
                     std::to_string(hw.sram_mb_per_core) + ", " +
                     std::to_string(hw.noc_width_bits) + ") (current DOCX single Core resource table)");
  }

  // --- current DOCX §1.2 outline/installed peaks; fixed 1 GHz ---
  const double width = static_cast<double>(hw.noc_width_bits);
  const double core_side_mm = (hw.sram_mb_per_core / 0.25) * 0.40662;
  hw.acore_mm2 = core_side_mm * core_side_mm;
  const long double side = static_cast<long double>(hw.mesh_n) * core_side_mm;
  const long double occupied = side * side;
  if (occupied > static_cast<long double>(hw.compute_area_mm2)) {
    detail::fail(path, "explicit mesh-nx x mesh-ny Core grid exceeds compute-area-mm2 at the DOCX SRAM-long-side Core outline; provide a legal hardware configuration, no automatic resize is performed");
  }
  if (hw.explicit_die_dimensions &&
      (side > static_cast<long double>(hw.die_width_mm) ||
       side > static_cast<long double>(hw.die_height_mm))) {
    detail::fail(path, "explicit Core grid does not fit die-width-mm and die-height-mm independently; no automatic hardware resize is performed");
  }
  // Fit above bounds the number of Cores for the legacy area's finite set.
  hw.cores = hw.mesh_n * hw.mesh_n;
  hw.pe_peak_die =
      static_cast<double>(hw.cores) * hw.core_row.pe_int4_gops_per_core * 1e9;
  hw.vec_peak_die =
      static_cast<double>(hw.cores) * hw.core_row.vec_g_eqop_per_core * 1e9;
  hw.sram_usable_bytes_per_core = 0.90 * hw.sram_mb_per_core * 1048576.0;
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
        config, {"schema_version", "profile", "die", "derived", "source_identity"}, path);
    if (config.contains("source_identity") &&
        config.at("source_identity") != source_identity_json()) {
      detail::fail(path + ".source_identity", "does not match the compiled Core32/DOCX source identity");
    }
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

  // Bare die node: explicit hardware keys including mesh-nx/mesh-ny, profile defaults to mesh_die_v1
  // (contract §3.2 default).
  return parse_die_node(config, path, profile);
}

// ---------------------------------------------------------------------------
// hardware_summary_json (contract §3.3)

nlohmann::json hardware_summary_json(const Hardware& hw) {
  nlohmann::json die;
  die["sram-mb-per-core"] = hw.sram_mb_per_core;
  die["mesh-nx"] = hw.mesh_n;
  die["mesh-ny"] = hw.mesh_n;
  die["compute-area-mm2"] = hw.compute_area_mm2;
  if (hw.explicit_die_dimensions) {
    die["die-width-mm"] = hw.die_width_mm;
    die["die-height-mm"] = hw.die_height_mm;
  }
  die["hbm-per-side"] = hw.hbm_per_side;
  die["hbm-sides"] = hw.hbm_sides;
  die["noc-width-bits"] = hw.noc_width_bits;

  nlohmann::json derived;
  derived["die_geometry_mode"] = hw.explicit_die_dimensions
      ? "explicit_width_height" : "legacy_area_only_square_compatibility";
  derived["mesh_n"] = hw.mesh_n;
  derived["cores"] = hw.cores;
  // The legacy key is kept for ABI consumers; the typed companion records
  // the actual INT4 unit and prevents this architectural peak being mistaken
  // for independently measured BF16/FP32 hardware throughput.
  derived["pe_peak_tflops"] = hw.pe_peak_die / 1e12;
  derived["pe_peak_tflops_deprecated_unit"] = "INT4 TOPS compatibility scalar; not BF16/FP32 TFLOPS";
  derived["pe_peak_int4_tops"] = hw.pe_peak_die / 1e12;
  derived["pe_peak_provenance"] = "DOCX architectural INT4 peak; legacy scalar model, not target hardware calibration";
  derived["vec_peak_teq"] = hw.vec_peak_die;
  derived["sram_usable_bytes_per_core"] = hw.sram_usable_bytes_per_core;
  derived["link_peak_bytes"] = hw.link_peak_bytes;
  derived["hbm_count"] = hw.hbm_count;
  derived["hbm_profile"] = "legacy_16GB_410GBps_per_stack_reference; not current DOCX HBM3E";
  derived["hbm_capacity_bytes"] = hw.hbm_capacity_bytes;
  derived["hbm_bandwidth_bytes"] = hw.hbm_bandwidth_bytes;

  nlohmann::json out;
  out["schema_version"] = 1;
  out["source_identity"] = source_identity_json();
  out["profile"] = to_string(hw.profile);
  out["die"] = die;
  out["derived"] = derived;
  return out;
}

}  // namespace DieModel
