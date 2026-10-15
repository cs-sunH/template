#include "test_support.hh"

#include <limits>
#include <string>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {

json explicit_input(const json& legacy) {
  json input = legacy;
  input.erase("search_profile");
  input.erase("validation_constraints");
  auto& architecture = input["architecture"];
  architecture["die_geometry"] = "explicit";
  architecture["die_width_mm"] = 21.0;
  architecture["die_height_mm"] = 29.0;
  architecture.erase("system_region_width_mm");
  architecture.erase("system_region_height_mm");
  input.erase("derived");
  return input;
}

void require_invalid(const json& input, const std::string& message) {
  bool rejected = false;
  try {
    (void)resolve_hardware(input);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, message);
}

void test_explicit_single_die_inputs_and_legacy_compatibility() {
  const auto legacy = read_json(std::string(COARSE_DIE_SOURCE_DIR) +
                                "/configs/example_hardware.json");
  const auto input = explicit_input(legacy);
  const auto inputBeforeResolve = input;
  const auto hardware = resolve_hardware(input);
  (void)resolve_hardware(input);
  require(input == inputBeforeResolve,
          "repeated resolution must not modify the caller's raw JSON");

  require(hardware.derived.at("core").at("N_core") == 88,
          "an explicit 8x11 grid must retain exactly 88 Cores");
  require(hardware.derived.at("die").at("core_columns_nx") == 8 &&
              hardware.derived.at("die").at("core_rows_ny") == 11,
          "the resolver must retain the user-specified rectangular grid");
  require(hardware.derived.at("die").at("width_mm") == 21.0 &&
              hardware.derived.at("die").at("height_mm") == 29.0 &&
              hardware.derived.at("die").at("area_mm2") == 609.0,
          "Die width, height, and area must come directly from explicit input");
  require(hardware.derived.at("hbm").at("total_stacks") == 4 &&
              hardware.derived.at("hbm").at("capacity_bytes") == 144.0e9,
          "explicit single-Die resolution must still derive the two HBM rows");
  require(hardware.derived.at("hbm").at("row_overhang") == true,
          "a legal HBM row overhang must be recorded rather than changing Die width");
  require(hardware.derived.at("system").at("status") == "not_requested",
          "omitting system-region fields must not require or invent Wafer placement");

  auto consistentScalarMirrors = input;
  consistentScalarMirrors["architecture"]["die_area_mm2"] = 609.0;
  consistentScalarMirrors["architecture"]["core_count"] = 88;
  const auto mirroredScalars = resolve_hardware(consistentScalarMirrors);
  require(mirroredScalars.derived.at("die") == hardware.derived.at("die") &&
              mirroredScalars.derived.at("core") == hardware.derived.at("core"),
          "consistent area/Core-count mirrors must be checked without overriding the inputs");
  auto mismatchedAreaMirror = consistentScalarMirrors;
  mismatchedAreaMirror["architecture"]["die_area_mm2"] = 610.0;
  require_invalid(mismatchedAreaMirror,
                  "a supplied Die area mirror must match width times height");
  auto mismatchedCoreMirror = consistentScalarMirrors;
  mismatchedCoreMirror["architecture"]["core_count"] = 89;
  require_invalid(mismatchedCoreMirror,
                  "a supplied Core-count mirror must match Nx times Ny");

  const double minWidth = hardware.derived.at("die").at("minimum_width_mm").get<double>();
  const double minHeight = hardware.derived.at("die").at("minimum_height_mm").get<double>();
  auto roomyInput = input;
  roomyInput["architecture"]["die_width_mm"] = minWidth + 3.125;
  roomyInput["architecture"]["die_height_mm"] = minHeight + 4.75;
  const auto roomy = resolve_hardware(roomyInput);
  require(roomy.derived.at("die").at("width_mm") == minWidth + 3.125 &&
              roomy.derived.at("die").at("height_mm") == minHeight + 4.75,
          "arbitrary legal Die clearances must be preserved exactly");
  require(roomy.derived.at("core").at("N_core") == 88 &&
              roomy.derived.at("die").at("core_count") == 88,
          "extra Die area must not add or rearrange Cores");

  auto inferredExplicit = input;
  inferredExplicit["architecture"].erase("die_geometry");
  const auto inferred = resolve_hardware(inferredExplicit);
  require(inferred.derived.at("die").at("width_mm") == 21.0 &&
              inferred.derived.at("die").at("height_mm") == 29.0,
          "W/H without a legacy geometry marker must be treated as explicit input");

  auto resolvedLegacyMarker = input;
  resolvedLegacyMarker["architecture"]["die_geometry"] = "resolved_source_boundary";
  resolvedLegacyMarker["architecture"]["geometry_candidate"] = "isotropic";
  const auto resolvedLegacy = resolve_hardware(resolvedLegacyMarker);
  require(resolvedLegacy.derived.at("die").at("width_mm") == 21.0 &&
              resolvedLegacy.derived.at("die").at("height_mm") == 29.0,
          "an explicit W/H must remain authoritative with a legacy resolved-geometry marker");

  auto legacySearchMetadata = input;
  legacySearchMetadata["search_profile"] = legacy.at("search_profile");
  legacySearchMetadata["search_profile"]["nx_range"] = {1, 1};
  legacySearchMetadata["search_profile"]["ny_range"] = {1, 1};
  legacySearchMetadata["search_profile"]["hbm_n_range"] = {1, 1};
  legacySearchMetadata["search_profile"]["geometry_search_mode"] = "wafer_first";
  legacySearchMetadata["search_profile"].erase("wafer_columns_range");
  legacySearchMetadata["search_profile"].erase("wafer_rows_range");
  const auto withMetadata = resolve_hardware(legacySearchMetadata);
  require(withMetadata.fingerprint == hardware.fingerprint &&
              withMetadata.derived.at("die") == hardware.derived.at("die") &&
              withMetadata.derived.at("core") == hardware.derived.at("core") &&
              withMetadata.rates == hardware.rates,
          "legacy search metadata must neither constrain nor alter an explicit instance");

  // Existing shipped configurations remain accepted through deterministic legacy
  // geometry interpretation. This compatibility path does not enumerate alternatives.
  const auto legacyHardware = resolve_hardware(legacy);
  require(legacyHardware.derived.at("die").at("geometry_candidate") == "minimum_grid" &&
              legacyHardware.derived.at("core").at("N_core") == 88,
          "the existing no-W/H example config must keep its legacy geometry interpretation");
  require(legacyHardware.derived.at("system").at("module_count").get<long long>() > 0,
          "legacy configurations that provide a system region may retain that diagnostic");
}

void test_optional_validation_and_invalid_hardware_inputs() {
  const auto legacy = read_json(std::string(COARSE_DIE_SOURCE_DIR) +
                                "/configs/example_hardware.json");
  const auto valid = explicit_input(legacy);
  const auto baseline = resolve_hardware(valid);

  auto validated = valid;
  validated["validation_constraints"] = {
      {"die_area_min_mm2", 608.0},
      {"die_area_max_mm2", 610.0},
      {"reticle_width_mm", 22.0},
      {"reticle_height_mm", 30.0},
      {"allow_reticle_rotation", false},
      {"die_max_aspect_ratio", 1.5},
      {"require_nonnegative_d2d_budget", true},
      {"require_hbm_row_within_die", false},
  };
  const auto validatedOriginal = validated;
  const auto validatedHardware = resolve_hardware(validated);
  require(validated == validatedOriginal,
          "successful instance validation must not rewrite the caller's configuration");
  require(validatedHardware.derived.at("die") == baseline.derived.at("die") &&
              validatedHardware.rates == baseline.rates &&
              validatedHardware.fingerprint == baseline.fingerprint,
          "passing validation constraints must not change the supplied instance or its identity");

  auto tooSmallByConstraint = valid;
  tooSmallByConstraint["validation_constraints"] = {{"die_area_min_mm2", 610.0}};
  const auto rejectedConstraintOriginal = tooSmallByConstraint;
  require_invalid(tooSmallByConstraint,
                  "validation_constraints must reject a failing current instance");
  require(tooSmallByConstraint == rejectedConstraintOriginal,
          "failed instance validation must leave the caller's configuration unchanged");

  auto unsupportedConstraint = valid;
  unsupportedConstraint["validation_constraints"] = {{"nx_range", {1, 4}}};
  require_invalid(unsupportedConstraint,
                  "validation_constraints must reject search-only or unknown keys");

  auto overflowingIngressWeights = valid;
  overflowingIngressWeights["performance_profile"]["timing_topology_profile"] =
      "explicit_ingress_v2";
  overflowingIngressWeights["performance_profile"]["explicit_ingress"] = {
      {"source", "hardware_input_test"},
      {"ingress_raw_bytes_per_s", {1.0e9, 1.0e9}},
      {"row_to_ingress", {0, 1}},
      {"shared_group_ids", {0, 1}},
      {"ingress_enabled", {true, true}},
      {"source_positions",
       {{{0, 1.0e308}, {1, 1.0e308}}, {{0, 1.0}}}},
  };
  require_invalid(overflowingIngressWeights,
                  "explicit ingress weights whose sum overflows must be rejected");

  auto halfSystemRegion = valid;
  halfSystemRegion["architecture"]["system_region_width_mm"] = 215.0;
  require_invalid(halfSystemRegion,
                  "a system region with only one dimension must be rejected");

  auto overflowingSystemRegion = valid;
  overflowingSystemRegion["architecture"]["system_region_width_mm"] =
      std::numeric_limits<double>::max();
  overflowingSystemRegion["architecture"]["system_region_height_mm"] =
      std::numeric_limits<double>::max();
  require_invalid(overflowingSystemRegion,
                  "system region dimensions that overflow module counts must be rejected");

  auto tooNarrow = valid;
  tooNarrow["architecture"]["die_width_mm"] =
      baseline.derived.at("die").at("minimum_width_mm").get<double>() - 1.0e-3;
  require_invalid(tooNarrow, "a Die narrower than its Core grid must be rejected");

  auto tooShort = valid;
  tooShort["architecture"]["die_height_mm"] =
      baseline.derived.at("die").at("minimum_height_mm").get<double>() - 1.0e-3;
  require_invalid(tooShort, "a Die shorter than its Core grid must be rejected");

  auto missingWidth = valid;
  missingWidth["architecture"].erase("die_width_mm");
  require_invalid(missingWidth, "explicit geometry requires a Die width");
  auto missingHeight = valid;
  missingHeight["architecture"].erase("die_height_mm");
  require_invalid(missingHeight, "explicit geometry requires a Die height");

  auto zeroWidth = valid;
  zeroWidth["architecture"]["die_width_mm"] = 0.0;
  require_invalid(zeroWidth, "Die width must be positive");
  auto nonfiniteHeight = valid;
  nonfiniteHeight["architecture"]["die_height_mm"] =
      std::numeric_limits<double>::infinity();
  require_invalid(nonfiniteHeight, "Die height must be finite");
  auto nanWidth = valid;
  nanWidth["architecture"]["die_width_mm"] =
      std::numeric_limits<double>::quiet_NaN();
  require_invalid(nanWidth, "Die width must reject NaN");

  auto zeroColumns = valid;
  zeroColumns["architecture"]["core_columns_nx"] = 0;
  require_invalid(zeroColumns, "Core column count must be positive");
  auto negativeRows = valid;
  negativeRows["architecture"]["core_rows_ny"] = -1;
  require_invalid(negativeRows, "Core row count must be positive");
  auto unsupportedGeometry = valid;
  unsupportedGeometry["architecture"]["die_geometry"] = "automatic_search";
  require_invalid(unsupportedGeometry,
                  "the resolver must reject an unsupported geometry marker instead of searching");

  auto missingTopology = valid;
  missingTopology["performance_profile"].erase("timing_topology_profile");
  bool stableTopologyError = false;
  try {
    (void)resolve_hardware(missingTopology);
  } catch (const std::invalid_argument& error) {
    stableTopologyError = std::string(error.what()).find("missing_topology_profile") !=
                          std::string::npos;
  }
  require(stableTopologyError,
          "missing explicit topology must retain the stable diagnostic");

  auto badConstant = valid;
  badConstant["source_constants"]["hbm_stack_peak_bytes_per_s"] = 1.0;
  require_invalid(badConstant,
                  "source constants must remain bound to immutable architecture values");
  auto badDigest = valid;
  badDigest["core_resource_table"]["rows_sha256"] =
      "8e7d393ff9a96dd556088fa7c98a6fd80ed0b277e03cde7c3909a0f118e45f37";
  require_invalid(badDigest,
                  "the compiled Core64 resource table digest must be validated");

  auto mirrored = valid;
  mirrored["derived"] = baseline.derived;
  const auto mirrorOriginal = mirrored;
  const auto acceptedMirror = resolve_hardware(mirrored);
  require(acceptedMirror.fingerprint == baseline.fingerprint,
          "a consistent derived mirror must not perturb canonical hardware identity");
  require(mirrored == mirrorOriginal,
          "resolving a consistent mirror must not mutate the caller's JSON");
  mirrored["derived"]["core"]["N_core"] = 89;
  const auto inconsistentMirrorOriginal = mirrored;
  require_invalid(mirrored,
                  "a client-supplied derived mirror is accepted only if consistent");
  require(mirrored == inconsistentMirrorOriginal,
          "rejecting an inconsistent mirror must not mutate the caller's JSON");
}

}  // namespace

int main() {
  return run_test("hardware_input_test", [] {
    test_explicit_single_die_inputs_and_legacy_compatibility();
    test_optional_validation_and_invalid_hardware_inputs();
  });
}
