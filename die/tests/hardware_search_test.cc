#include "test_support.hh"

#include <algorithm>
#include <limits>
#include <set>
#include <tuple>
#include <vector>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {

json make_search_input(const json& base, int nx_min, int nx_max, int ny_min,
                       int ny_max, int n_min, int n_max,
                       const std::string& mode) {
  json input = base;
  auto& search = input["search_profile"];
  search["nx_range"] = {nx_min, nx_max};
  search["ny_range"] = {ny_min, ny_max};
  search["hbm_n_range"] = {n_min, n_max};
  search["geometry_search_mode"] = mode;
  input["architecture"]["system_region_height_mm"] = 160.0;
  return input;
}

using Identity = std::tuple<int, int, int, int, int>;

Identity physical_identity(const json& candidate) {
  const auto& id = candidate.at("candidate_identity");
  return {id.at("core_columns_nx").get<int>(),
          id.at("core_rows_ny").get<int>(),
          id.at("hbm_stacks_per_row").get<int>(),
          id.at("wafer_columns").get<int>(),
          id.at("wafer_rows").get<int>()};
}

json only_candidate(json input, int nx, int ny, int hbmN) {
  auto& search = input["search_profile"];
  search["nx_range"] = {nx, nx};
  search["ny_range"] = {ny, ny};
  search["hbm_n_range"] = {hbmN, hbmN};
  return search_hardware(input);
}

void test_multicandidate_search_and_input_validation() {
  const auto base = read_json(std::string(COARSE_DIE_SOURCE_DIR) +
                              "/configs/example_hardware.json");
  auto coreInput = make_search_input(base, 8, 9, 10, 11, 1, 2, "core_first");

  std::vector<Identity> expected;
  int minColumns = std::numeric_limits<int>::max();
  int maxColumns = 0;
  int minRows = std::numeric_limits<int>::max();
  int maxRows = 0;
  for (int nx = 8; nx <= 9; ++nx) {
    for (int ny = 10; ny <= 11; ++ny) {
      for (int n = 1; n <= 2; ++n) {
        const auto one = only_candidate(coreInput, nx, ny, n);
        require(one.at("candidate_count").get<std::size_t>() == 1,
                "single Nx/Ny/n choice must identify one source geometry candidate");
        const auto& best = one.at("best");
        const auto& identity = best.at("candidate_identity");
        const int columns = best.at("derived").at("system").at("module_columns").get<int>();
        const int rows = best.at("derived").at("system").at("module_rows").get<int>();
        minColumns = std::min(minColumns, columns);
        maxColumns = std::max(maxColumns, columns);
        minRows = std::min(minRows, rows);
        maxRows = std::max(maxRows, rows);
        expected.emplace_back(nx, ny, n, columns, rows);
        require(identity.at("geometry") == "minimum_grid",
                "selected test grid must exercise minimum_grid source geometry");
      }
    }
  }
  require(expected.size() > 1 && minColumns < maxColumns && minRows < maxRows,
          "the search fixture must span multiple Core and wafer candidates");

  auto& boundedSearchProfile = coreInput["search_profile"];
  boundedSearchProfile["wafer_columns_range"] = {minColumns, maxColumns};
  boundedSearchProfile["wafer_rows_range"] = {minRows, maxRows};
  auto waferInput = coreInput;
  waferInput["search_profile"]["geometry_search_mode"] = "wafer_first";

  const auto coreSearch = search_hardware(coreInput);
  const auto waferSearch = search_hardware(waferInput);
  require(coreSearch.at("candidate_count") == expected.size() &&
              waferSearch.at("candidate_count") == expected.size(),
          "Core-first and Wafer-first must retain the same multi-candidate feasible set");
  std::set<Identity> expectedSet(expected.begin(), expected.end());
  std::set<Identity> observedCore;
  std::set<Identity> observedWafer;
  for (int nx = 8; nx <= 9; ++nx) {
    for (int ny = 10; ny <= 11; ++ny) {
      for (int n = 1; n <= 2; ++n) {
        const auto coreOne = only_candidate(coreInput, nx, ny, n);
        const auto waferOne = only_candidate(waferInput, nx, ny, n);
        require(coreOne.at("candidate_count") == waferOne.at("candidate_count"),
                "both search modes must agree on per-configuration feasibility");
        const auto& coreBest = coreOne.at("best");
        const auto& waferBest = waferOne.at("best");
        if (coreOne.at("candidate_count").get<std::size_t>() == 0) continue;
        const auto coreIdentity = physical_identity(coreBest);
        const auto waferIdentity = physical_identity(waferBest);
        require(coreIdentity == waferIdentity,
                "both search modes must identify the same Nx/Ny/HBM/Mx/My tuple");
        observedCore.insert(coreIdentity);
        observedWafer.insert(waferIdentity);
        for (const char* section : {"core", "die", "hbm", "system", "hardware_summary", "resources"})
          require(coreBest.at("derived").at(section) == waferBest.at("derived").at(section),
                  std::string("search mode changed derived section ") + section);
        require(coreBest.at("rates") == waferBest.at("rates") &&
                    coreBest.at("fingerprint") == waferBest.at("fingerprint"),
                "search mode must preserve rates and hardware fingerprint");
      }
    }
  }
  require(observedCore == expectedSet && observedWafer == expectedSet,
          "both generators must cover every independently queried feasible tuple");

  auto bounded = waferInput;
  bounded["search_profile"]["wafer_columns_range"] = {minColumns, minColumns};
  const auto boundedSearch = search_hardware(bounded);
  require(boundedSearch.at("candidate_count").get<std::size_t>() < expected.size(),
          "wafer_columns_range upper bound must filter larger derived column counts");
  bounded["search_profile"]["wafer_columns_range"] = {maxColumns + 1, maxColumns + 1};
  bool lowerBoundConsumed = false;
  try {
    (void)search_hardware(bounded);
  } catch (const std::invalid_argument&) {
    lowerBoundConsumed = true;
  }
  require(lowerBoundConsumed, "wafer_columns_range lower bound must reject out-of-range candidates");

  auto arbitraryGeometry = base;
  const auto ordinary = resolve_hardware(base);
  arbitraryGeometry["architecture"]["die_width_mm"] =
      ordinary.derived.at("die").at("width_mm").get<double>() + 0.01;
  bool arbitraryRejected = false;
  try {
    (void)resolve_hardware(arbitraryGeometry);
  } catch (const std::invalid_argument&) {
    arbitraryRejected = true;
  }
  require(arbitraryRejected,
          "standard source geometry must reject arbitrary Die width expansion");

  auto mirror = base;
  mirror["derived"] = ordinary.derived;
  const auto mirrored = resolve_hardware(mirror);
  require(mirrored.fingerprint == ordinary.fingerprint,
          "a consistent derived mirror must not perturb canonical hardware identity");
  mirror["derived"]["core"]["N_core"] = 89;
  bool inconsistentMirrorRejected = false;
  try {
    (void)resolve_hardware(mirror);
  } catch (const std::invalid_argument&) {
    inconsistentMirrorRejected = true;
  }
  require(inconsistentMirrorRejected,
          "a client-supplied derived mirror may only be accepted when consistent");

  auto missingTopology = base;
  missingTopology["performance_profile"].erase("timing_topology_profile");
  bool explicitTopologyDiagnostic = false;
  try {
    (void)resolve_hardware(missingTopology);
  } catch (const std::invalid_argument& e) {
    explicitTopologyDiagnostic = std::string(e.what()).find("missing_topology_profile") !=
                                 std::string::npos;
  }
  require(explicitTopologyDiagnostic,
          "missing explicit topology must carry the stable missing_topology_profile diagnostic");

  auto badConstant = base;
  badConstant["source_constants"]["hbm_stack_peak_bytes_per_s"] = 1.0;
  bool constantRejected = false;
  try {
    (void)resolve_hardware(badConstant);
  } catch (const std::invalid_argument&) {
    constantRejected = true;
  }
  require(constantRejected, "source constant mirror must stay bound to immutable source values");
  auto badDigest = base;
  badDigest["core_resource_table"]["rows_sha256"] =
      "8e7d393ff9a96dd556088fa7c98a6fd80ed0b277e03cde7c3909a0f118e45f37";
  bool digestRejected = false;
  try {
    (void)resolve_hardware(badDigest);
  } catch (const std::invalid_argument&) {
    digestRejected = true;
  }
  require(digestRejected, "source Core64 row digest must be validated against compiled row content");

  require(ordinary.derived.at("hbm").at("row_to_ingress") == json::array({nullptr, nullptr}),
          "distributed topology must not invent a source-defined row ingress mapping");
}

}  // namespace

int main() {
  return run_test("hardware_search_test", test_multicandidate_search_and_input_validation);
}
