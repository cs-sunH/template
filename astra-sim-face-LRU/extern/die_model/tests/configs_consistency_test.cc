/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
configs_consistency_test.cc -- anti-drift check between the code constants
and configs/*.json (contract §1: the JSON files are the documented source of
truth; the library embeds the same values and this test forces both sides to
stay identical):

  configs/core_table.json                 == core_table() (14 rows);
  configs/calibration_mesh_die_v1.json    == the mesh profile coefficients
                                            of make_hardware();
  configs/calibration_tx8_reference_v1.json == the TX8 reference constants
                                            of make_hardware();
  configs/VERSION                          == a die_model/<version> string.

Doubles are compared with exact equality on purpose: both sides are produced
by the same §2.1 formulas / §3.1 literals, so any difference is real drift.
The configs directory is injected by CMake (DIE_MODEL_CONFIGS_DIR).

Build (standalone): cmake -S extern/die_model -B <build> && cmake --build
<build> --target configs_consistency_test.  Exit code 0 on ALL PASS.
*/

#include "DieModel.hh"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool g_ok = true;

void expect(bool cond, const char* what) {
  if (!cond) {
    std::fprintf(stderr, "[configs_consistency_test] FAIL: %s\n", what);
    g_ok = false;
  }
}

void expect_double(double actual, double expected, const char* what) {
  if (actual != expected) {
    std::fprintf(stderr,
                 "[configs_consistency_test] FAIL: %s: code %.17g != json "
                 "%.17g\n",
                 what, actual, expected);
    g_ok = false;
  }
}

nlohmann::json read_json(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "[configs_consistency_test] FAIL: cannot open %s\n",
                 path.c_str());
    g_ok = false;
    return nlohmann::json();
  }
  try {
    return nlohmann::json::parse(in);
  } catch (const std::exception& e) {
    std::fprintf(stderr,
                 "[configs_consistency_test] FAIL: parse %s: %s\n",
                 path.c_str(), e.what());
    g_ok = false;
    return nlohmann::json();
  }
}

}  // namespace

int main() {
  const std::string dir = DIE_MODEL_CONFIGS_DIR;

  // --- core table: 14 rows, same order, same values ---
  const nlohmann::json table = read_json(dir + "/core_table.json");
  const std::vector<DieModel::CoreTableEntry> rows = DieModel::core_table();
  expect(table.at("rows").size() == 14, "core table has 14 rows");
  expect(rows.size() == 14, "core_table() returns 14 rows");
  for (std::size_t i = 0; i < rows.size() && i < table.at("rows").size();
       ++i) {
    const nlohmann::json& r = table.at("rows")[i];
    const std::string what = "core_table row " + std::to_string(i);
    expect_double(r.at("sram-mb-per-core").get<double>(),
                  rows[i].sram_mb_per_core, (what + " sram").c_str());
    expect(r.at("core-type").get<std::string>() ==
               DieModel::to_string(rows[i].core_type),
           (what + " type").c_str());
    expect_double(r.at("vec-g-eqop-per-core").get<double>(),
                  rows[i].vec_g_eqop_per_core, (what + " vec").c_str());
    expect_double(r.at("pe-gflops-per-core").get<double>(),
                  rows[i].pe_gflops_per_core, (what + " pe").c_str());
    expect_double(r.at("side-ref-um").get<double>(), rows[i].core_side_ref_um,
                  (what + " side").c_str());
  }

  // A Hardware to carry the calibration coefficients of both profiles
  // (HBM 3x2 keeps area 800 inside the design doc §3.1 layout table).
  nlohmann::json die;
  die["sram-mb-per-core"] = 0.75;
  die["core-type"] = "large";
  die["compute-area-mm2"] = 800;
  die["hbm-per-side"] = 3u;
  die["hbm-sides"] = 2u;
  die["noc-width-bits"] = 512u;
  nlohmann::json config;
  // Unsigned literals: the strict parser requires unsigned integer storage.
  config["schema_version"] = 1u;
  config["profile"] = "mesh_die_v1";
  config["die"] = die;
  const DieModel::Hardware hw = DieModel::make_hardware(config);

  // --- mesh_die_v1 coefficients (§6.1 defaults; §2.1 source) ---
  const nlohmann::json mesh = read_json(dir + "/calibration_mesh_die_v1.json");
  expect(mesh.at("profile").get<std::string>() == "mesh_die_v1",
         "mesh calibration profile tag");
  expect_double(hw.eta_pe, mesh.at("eta-pe").get<double>(), "eta-pe");
  expect_double(hw.eta_vec, mesh.at("eta-vec").get<double>(), "eta-vec");
  expect_double(hw.eta_matrix_mem, mesh.at("eta-matrix-mem").get<double>(),
                "eta-matrix-mem");
  expect_double(hw.eta_vector_mem, mesh.at("eta-vector-mem").get<double>(),
                "eta-vector-mem");
  expect_double(hw.eta_noc, mesh.at("eta-noc").get<double>(), "eta-noc");
  expect_double(hw.launch_us_per_macro_group,
                mesh.at("launch-us-per-macro-group").get<double>(),
                "launch-us-per-macro-group");

  // --- tx8_reference_v1 constants (§2.1) ---
  const nlohmann::json tx8 =
      read_json(dir + "/calibration_tx8_reference_v1.json");
  expect(tx8.at("profile").get<std::string>() == "tx8_reference_v1",
         "tx8 calibration profile tag");
  expect_double(hw.tx8_matrix_peak_flop_s,
                tx8.at("matrix-peak-flop-per-s").get<double>(),
                "matrix-peak-flop-per-s");
  expect_double(hw.tx8_vector_peak_eqop_s,
                tx8.at("vector-peak-eqop-per-s").get<double>(),
                "vector-peak-eqop-per-s");
  expect_double(hw.tx8_bw_matrix_bytes_s,
                tx8.at("bw-matrix-bytes-per-s").get<double>(),
                "bw-matrix-bytes-per-s");
  expect_double(hw.tx8_bw_vector_bytes_s,
                tx8.at("bw-vector-bytes-per-s").get<double>(),
                "bw-vector-bytes-per-s");
  expect_double(hw.tx8_extra_launch_s, tx8.at("extra-launch-s").get<double>(),
                "extra-launch-s");
  expect_double(hw.tx8_extra_memory_latency_s,
                tx8.at("extra-memory-latency-s").get<double>(),
                "extra-memory-latency-s");
  expect(hw.full_score_materialization ==
             tx8.at("full-score-materialization").get<bool>(),
         "full-score-materialization");
  expect(hw.skip_fully_masked_tiles ==
             tx8.at("skip-fully-masked-tiles").get<bool>(),
         "skip-fully-masked-tiles");

  // --- VERSION ---
  {
    std::ifstream in(dir + "/VERSION");
    std::string version;
    std::getline(in, version);
    expect(in && version.rfind("die_model/", 0) == 0,
           "VERSION carries die_model/<semver>");
  }

  if (g_ok) {
    std::printf("[configs_consistency_test] ALL PASS\n");
    return 0;
  }
  return 1;
}
