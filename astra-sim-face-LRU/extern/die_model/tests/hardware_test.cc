/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
hardware_test.cc -- die hardware derivation (design doc §3.1; contract
§3.1/§3.3).

Layout legality follows the design doc §3.1 ten-combination table (the
contract §3.1 text delegates to it: "合法组合见【设计文档】§3.1").  Note for
the stage A1 report: the contract's §3.1 example JSON / §3.3 derived example
/ §3.5 face default use (800 mm2, 2 per side, 2 sides), which is NOT in that
table (800 mm2 pairs with per-side 3/4 only); this test pins the design-doc
table, including rejecting (800,2,2), until the orchestrator rules
otherwise.

  (0.75 MB, large, 800 mm2, 512 bit, HBM 3x2) -> the §3.1 example geometry:
      mesh_n 35, cores 1225, PE peak 1225*1218e9 FLOP/s (1492.05 TFLOPS),
      vec peak 152613562500000 eq-op/s, SRAM usable 675000 bytes/core, link
      peak 64e9 byte/s, HBM count 6 / 96e9 byte / 2460e9 byte/s;
  (1 MB, large, 800 mm2, 1024 bit, HBM 4x1) -> fewer cores than the example
      point even with more SRAM (wider NoC grows Acore): mesh_n 24, cores
      576, PE 576*2700e9, vec 576*249.165e9, HBM count 4 / 64e9 / 1640e9;
  HBM (2 per side, 2 sides) on a legal 500 mm2 die -> count 4, capacity
      64e9, bandwidth 1640e9 (the per-stack 16e9/410e9 constants);
  illegal inputs throw DieModelError: layouts (300,3,1), (800,5,1) and
      (800,2,2), NoC width 2048, (sram=3, small) off the core table, bad
      core-type string, area 700;
  hardware_summary_json carries every contract §3.3 key with the right
      values, and the wrapped form accepts (and ignores) its own "derived"
      mirror so resolved files round-trip.

Build (standalone): cmake -S extern/die_model -B <build> && cmake --build
<build> --target hardware_test.  Exit code 0 on ALL PASS.
*/

#include "DieModel.hh"

#include <cmath>
#include <cstdio>
#include <string>

namespace {

bool g_ok = true;

void expect(bool cond, const char* what) {
  if (!cond) {
    std::fprintf(stderr, "[hardware_test] FAIL: %s\n", what);
    g_ok = false;
  }
}

void expect_near(double actual, double expected, double rel_tol,
                 const char* what) {
  if (!(std::abs(actual - expected) <= rel_tol * std::abs(expected))) {
    std::fprintf(stderr,
                 "[hardware_test] FAIL: %s = %.17g, expected %.17g (rel err "
                 "%.3e > %.3e)\n",
                 what, actual, expected,
                 std::abs(actual - expected) / std::abs(expected), rel_tol);
    g_ok = false;
  }
}

void expect_uint(std::uint64_t actual, std::uint64_t expected,
                 const char* what) {
  if (actual != expected) {
    std::fprintf(stderr,
                 "[hardware_test] FAIL: %s = %llu, expected %llu\n", what,
                 static_cast<unsigned long long>(actual),
                 static_cast<unsigned long long>(expected));
    g_ok = false;
  }
}

nlohmann::json die_node(double sram, const char* type, double area,
                        std::uint64_t per_side, std::uint64_t sides,
                        std::uint64_t width) {
  nlohmann::json die;
  die["sram-mb-per-core"] = sram;
  die["core-type"] = type;
  die["compute-area-mm2"] = area;
  die["hbm-per-side"] = per_side;
  die["hbm-sides"] = sides;
  die["noc-width-bits"] = width;
  return die;
}

nlohmann::json wrapped(const nlohmann::json& die, const char* profile) {
  nlohmann::json j;
  // Unsigned literal: the strict parser requires unsigned integer storage.
  j["schema_version"] = 1u;
  j["profile"] = profile;
  j["die"] = die;
  return j;
}

// Runs make_hardware and expects DieModelError; optionally checks that the
// message contains the given substring (field path / key name).
void expect_reject(const nlohmann::json& j, const char* what,
                   const char* message_contains = nullptr) {
  try {
    DieModel::make_hardware(j);
    std::fprintf(stderr, "[hardware_test] FAIL: %s (accepted, expected "
                         "DieModelError)\n",
                 what);
    g_ok = false;
  } catch (const DieModel::DieModelError& e) {
    if (message_contains &&
        std::string(e.what()).find(message_contains) == std::string::npos) {
      std::fprintf(stderr,
                   "[hardware_test] FAIL: %s (message \"%s\" lacks \"%s\")\n",
                   what, e.what(), message_contains);
      g_ok = false;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr,
                 "[hardware_test] FAIL: %s (wrong exception type: %s)\n", what,
                 e.what());
    g_ok = false;
  }
}

}  // namespace

int main() {
  using DieModel::make_hardware;

  // --- example point (design doc §3.1 geometry; HBM 3x2 keeps area 800
  //     inside the ten-combination table) ---
  const DieModel::Hardware hw = make_hardware(
      wrapped(die_node(0.75, "large", 800, 3, 2, 512), "mesh_die_v1"));
  expect_uint(hw.mesh_n, 35, "mesh_n 35");
  expect_uint(hw.cores, 1225, "cores 1225");
  expect_near(hw.pe_peak_die, 1225.0 * 1218e9, 1e-9, "PE peak 1225*1218e9");
  expect_near(hw.pe_peak_die / 1e12, 1492.05, 1e-9, "PE peak 1492.05 TFLOPS");
  expect_near(hw.vec_peak_die, 152613562500000.0, 1e-9, "vec peak");
  expect_near(hw.sram_usable_bytes_per_core, 675000.0, 1e-9, "SRAM usable");
  expect_near(hw.link_peak_bytes, 64e9, 1e-9, "link peak 64e9");
  expect_uint(hw.hbm_count, 6, "HBM count 6");
  expect_near(hw.hbm_capacity_bytes, 96e9, 1e-9, "HBM capacity 96e9");
  expect_near(hw.hbm_bandwidth_bytes, 2460e9, 1e-9, "HBM bandwidth 2460e9");
  // Acore for large/512bit = (787.260242um/1000)^2 * (0.85+0.15*1).
  expect_near(hw.acore_mm2, 0.6197786886338985, 1e-15, "Acore");
  // Cached core table row (§3.1 table row 0.75/large).
  expect_near(hw.core_row.vec_g_eqop_per_core, 124.5825, 1e-15,
              "core row vec");
  expect_near(hw.core_row.pe_gflops_per_core, 1218, 1e-15, "core row PE");
  expect_near(hw.core_row.core_side_ref_um, 787.260242, 1e-15,
              "core row side");
  // Profile coefficients (§6.1 defaults, §2.1 source).
  expect_near(hw.eta_pe, 0.9079551923877489, 0.0, "eta_pe");
  expect_near(hw.eta_vec, 0.9079551923877489, 0.0, "eta_vec");
  expect_near(hw.eta_matrix_mem, 0.70, 0.0, "eta_matrix_mem");
  expect_near(hw.eta_vector_mem, 0.34116682926829267, 0.0, "eta_vector_mem");
  expect_near(hw.eta_noc, 0.70, 0.0, "eta_noc");
  expect_near(hw.launch_us_per_macro_group, 1.0, 0.0, "launch_us");

  // --- (1 MB, large, 800 mm2, 1024 bit): wider NoC, fewer cores ---
  const DieModel::Hardware hw2 = make_hardware(
      wrapped(die_node(1.0, "large", 800, 4, 1, 1024), "mesh_die_v1"));
  // Acore = (1076.754654um/1000)^2 * (0.85 + 0.15*1024/512) = 1.333310672...
  expect_near(hw2.acore_mm2, 1.333310672647259, 1e-15, "Acore 1MB/1024bit");
  expect_uint(hw2.mesh_n, 24, "mesh_n 24");
  expect_uint(hw2.cores, 576, "cores 576");
  expect_near(hw2.pe_peak_die, 576.0 * 2700e9, 1e-9, "PE peak 576*2700e9");
  expect_near(hw2.vec_peak_die, 576.0 * 249.165e9, 1e-9, "vec peak 1MB");
  expect_near(hw2.sram_usable_bytes_per_core, 900000.0, 1e-9, "SRAM 1MB");
  expect_near(hw2.link_peak_bytes, 128e9, 1e-9, "link peak 128e9");
  expect_uint(hw2.hbm_count, 4, "HBM count 4 (4x1)");
  expect_near(hw2.hbm_capacity_bytes, 64e9, 1e-9, "HBM capacity 64e9 (4x1)");
  expect_near(hw2.hbm_bandwidth_bytes, 1640e9, 1e-9,
              "HBM bandwidth 1640e9 (4x1)");
  expect(hw2.cores < hw.cores, "1MB/1024bit has fewer cores than 0.75/512");

  // --- HBM 2 per side x 2 sides on a legal 500 mm2 die: 4 stacks, 64e9
  //     byte, 1640e9 byte/s (per-stack 16e9/410e9, design doc §3.1) ---
  const DieModel::Hardware hw5 = make_hardware(
      wrapped(die_node(0.75, "large", 500, 2, 2, 256), "mesh_die_v1"));
  expect_uint(hw5.hbm_count, 4, "HBM count 4 (2x2)");
  expect_near(hw5.hbm_capacity_bytes, 64e9, 1e-9, "HBM capacity 64e9 (2x2)");
  expect_near(hw5.hbm_bandwidth_bytes, 1640e9, 1e-9,
              "HBM bandwidth 1640e9 (2x2)");

  // --- bare die node form; profile defaults to mesh_die_v1 ---
  const DieModel::Hardware hw3 =
      make_hardware(die_node(0.75, "large", 800, 3, 2, 512));
  expect_uint(hw3.cores, 1225, "bare form cores");
  expect(hw3.profile == DieModel::Profile::MeshDieV1, "default profile mesh");

  // --- tx8 profile accepted; wrapped form with own derived mirror OK ---
  const DieModel::Hardware hw4 = make_hardware(
      wrapped(die_node(0.75, "large", 800, 3, 2, 512), "tx8_reference_v1"));
  expect(hw4.profile == DieModel::Profile::Tx8ReferenceV1, "tx8 profile");
  nlohmann::json round_trip = wrapped(die_node(0.75, "large", 800, 3, 2, 512),
                                      "mesh_die_v1");
  round_trip["derived"] = DieModel::hardware_summary_json(hw)["derived"];
  expect_uint(make_hardware(round_trip).cores, 1225,
              "resolved file round-trip");

  // --- illegal inputs (design doc §3.1 legal set) ---
  expect_reject(wrapped(die_node(0.75, "large", 300, 3, 1, 512),
                        "mesh_die_v1"),
                "layout (300,3,1)", "illegal");
  expect_reject(wrapped(die_node(0.75, "large", 800, 5, 1, 512),
                        "mesh_die_v1"),
                "layout (800,5,1)");
  // (800,2,2): used by the contract §3.1/§3.3/§3.5 examples but absent from
  // the design doc §3.1 table; rejected here pending the orchestrator's
  // ruling (see the stage A1 report).
  expect_reject(wrapped(die_node(0.75, "large", 800, 2, 2, 512),
                        "mesh_die_v1"),
                "layout (800,2,2) not in §3.1 table");
  expect_reject(wrapped(die_node(0.75, "large", 800, 3, 2, 2048),
                        "mesh_die_v1"),
                "NoC width 2048", "noc-width-bits");
  expect_reject(wrapped(die_node(3.0, "small", 800, 3, 2, 256),
                        "mesh_die_v1"),
                "(sram=3,small) off core table", "core table");
  expect_reject(wrapped(die_node(0.75, "medium", 800, 3, 2, 256),
                        "mesh_die_v1"),
                "core-type medium");
  expect_reject(wrapped(die_node(0.75, "large", 700, 3, 2, 256),
                        "mesh_die_v1"),
                "area 700");
  {
    nlohmann::json missing_key =
        wrapped(die_node(0.75, "large", 800, 3, 2, 512), "mesh_die_v1");
    missing_key["die"].erase("noc-width-bits");
    expect_reject(missing_key, "missing noc-width-bits", "noc-width-bits");
  }
  {
    nlohmann::json unknown_key =
        wrapped(die_node(0.75, "large", 800, 3, 2, 512), "mesh_die_v1");
    unknown_key["die"]["turbo-mode"] = true;
    expect_reject(unknown_key, "unknown die key turbo-mode", "turbo-mode");
  }
  expect_reject(wrapped(die_node(0.75, "large", 800, 3, 2, 512),
                        "mesh_die_v9"),
                "unknown profile");

  // --- hardware_summary_json (contract §3.3) ---
  const nlohmann::json summary = DieModel::hardware_summary_json(hw);
  expect(summary.value("schema_version", 0) == 1, "summary schema_version");
  expect(summary.value("profile", "") == "mesh_die_v1", "summary profile");
  expect(summary.at("die").value("sram-mb-per-core", 0.0) == 0.75,
         "summary die sram");
  expect(summary.at("die").value("core-type", "") == "large",
         "summary die type");
  expect(summary.at("die").value("compute-area-mm2", 0.0) == 800.0,
         "summary die area");
  expect(summary.at("die").value("hbm-per-side", 0) == 3, "summary die ps");
  expect(summary.at("die").value("hbm-sides", 0) == 2, "summary die sides");
  expect(summary.at("die").value("noc-width-bits", 0) == 512,
         "summary die width");
  const nlohmann::json& dv = summary.at("derived");
  expect_uint(dv.value("mesh_n", 0), 35, "summary mesh_n");
  expect_uint(dv.value("cores", 0), 1225, "summary cores");
  expect_near(dv.value("pe_peak_tflops", -1.0), 1492.05, 1e-9,
              "summary pe_peak_tflops");
  expect_near(dv.value("vec_peak_teq", -1.0), 152613562500000.0, 1e-9,
              "summary vec_peak_teq");
  expect_near(dv.value("sram_usable_bytes_per_core", -1.0), 675000.0, 1e-9,
              "summary sram usable");
  expect_near(dv.value("link_peak_bytes", -1.0), 64e9, 1e-9,
              "summary link peak");
  expect_uint(dv.value("hbm_count", 0), 6, "summary hbm_count");
  expect_near(dv.value("hbm_capacity_bytes", -1.0), 96e9, 1e-9,
              "summary hbm capacity");
  expect_near(dv.value("hbm_bandwidth_bytes", -1.0), 2460e9, 1e-9,
              "summary hbm bandwidth");

  if (g_ok) {
    std::printf("[hardware_test] ALL PASS\n");
    return 0;
  }
  return 1;
}
