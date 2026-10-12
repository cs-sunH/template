/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
tx8_reference_test.cc -- the TX8 reference standard (design doc §2).

Checks the floating-point cost core against the §2.3 reference table: all 13
timed groups, the attention-core subtotal (groups 2-9) and the block total
(groups 1-13) within 1e-6 us absolute per group and per total; the FFN
intra-group accounting (gate/up/down, SiLU, gate_mul); the FFN reference
traffic and external-memory service budget; and the F_tile=512 working-set
capacity check.  Expected values are hardcoded from design doc §2.3 (the
workbook 计时组13行 numbers); the library computes everything from the §2.2
formulas with no time lookup.

Build (standalone): cmake -S extern/die_model -B <build> && cmake --build
<build> --target tx8_reference_test.  Exit code 0 on ALL PASS.
*/

#include "DieModel.hh"

#include <cmath>
#include <cstdio>

namespace {

bool g_ok = true;

void expect(bool cond, const char* what) {
  if (!cond) {
    std::fprintf(stderr, "[tx8_reference_test] FAIL: %s\n", what);
    g_ok = false;
  }
}

// |actual - expected| < 1e-6 (microsecond-scale checks, design doc §2.3).
void expect_us(double actual, double expected, const char* what) {
  if (!(std::abs(actual - expected) < 1e-6)) {
    std::fprintf(stderr,
                 "[tx8_reference_test] FAIL: %s = %.9f us, expected %.9f us "
                 "(abs err %.3e >= 1e-6)\n",
                 what, actual, expected, std::abs(actual - expected));
    g_ok = false;
  }
}

void expect_exact(double actual, double expected, const char* what) {
  if (actual != expected) {
    std::fprintf(stderr,
                 "[tx8_reference_test] FAIL: %s = %.17g, expected %.17g\n",
                 what, actual, expected);
    g_ok = false;
  }
}

}  // namespace

int main() {
  const DieModel::Tx8Result r = DieModel::tx8_reference_block();

  expect(r.groups.size() == 13, "13 timed groups");

  // §2.3 table, in order; expected group totals in microseconds.
  struct Expected {
    const char* name;
    double us;
    const char* mem_class;
  };
  const Expected expected[13] = {
      {"RMSNorm_1", 246.000000, "vector"},
      {"Q_proj", 359.511771, "matrix"},
      {"K_proj", 134.816914, "matrix"},
      {"V_proj", 134.816914, "matrix"},
      {"RoPE_QK", 315.033675, "vector"},
      {"QKT", 1033.596343, "matrix"},
      {"score_mul_softmax", 3934.079063, "vector"},
      {"AV/PV", 1033.596343, "matrix"},
      {"O_proj", 359.511771, "matrix"},
      {"res_add_1", 368.819912, "vector"},
      {"RMSNorm_2", 246.000000, "vector"},
      {"FFN", 3363.000000, "matrix"},
      {"res_add_2", 368.819912, "vector"},
  };
  for (std::size_t i = 0; i < 13; ++i) {
    const DieModel::Tx8GroupTiming& g = r.groups[i];
    expect(std::string(g.name) == expected[i].name, "group name in order");
    expect(std::string(g.mem_class) == expected[i].mem_class,
           "group memory class");
    // Group completes at max(compute branch, memory stream), design doc §2.2.
    expect(g.total_s == (g.compute_s > g.mem_s ? g.compute_s : g.mem_s),
           "total = max(compute, mem)");
    expect_us(g.total_s * 1e6, expected[i].us, expected[i].name);
  }

  // Subtotals (design doc §2.3): attention core = groups 2-9, block = 1-13.
  expect_us(r.attention_core_s * 1e6, 7304.962795, "attention core 2-9");
  expect_us(r.block_total_s * 1e6, 11897.602619, "block total 1-13");

  // FFN intra-group accounting (design doc §2.3): compute-resource split,
  // five values that sum to the FFN compute budget, not extra kernels.
  expect_us(r.ffn_gate_s * 1e6, 1034.769230769, "FFN gate");
  expect_us(r.ffn_up_s * 1e6, 1034.769230769, "FFN up");
  expect_us(r.ffn_down_s * 1e6, 1034.769230769, "FFN down");
  expect_us(r.ffn_silu_s * 1e6, 242.524038462, "FFN SiLU");
  expect_us(r.ffn_gate_mul_s * 1e6, 16.168269231, "FFN gate_mul");

  // FFN reference traffic and external-memory service budget (§2.3): the
  // budget is below the 3363 us compute budget.
  expect_exact(r.ffn_traffic_bytes, 369098752.0, "FFN traffic bytes");
  expect_us(r.ffn_memory_budget_s * 1e6, 2636.419657, "FFN memory budget");
  expect(r.ffn_memory_budget_s < 3363e-6, "FFN memory < compute budget");

  // Reference capacity check with the workbook F_tile = 512 (§2.3).
  expect_exact(r.ffn_working_set_bytes, 40894464.0, "F_tile=512 working set");
  expect_exact(r.sram_capacity_limit_bytes, 0.9 * 48e6, "0.9*48e6 limit");
  expect(r.sram_capacity_ok, "working set fits 0.9*48e6");

  if (g_ok) {
    std::printf("[tx8_reference_test] ALL PASS (13 groups, attention %.6f us,"
                " block %.6f us)\n",
                r.attention_core_s * 1e6, r.block_total_s * 1e6);
    return 0;
  }
  return 1;
}
