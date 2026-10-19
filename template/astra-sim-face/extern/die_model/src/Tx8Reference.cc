/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
Tx8Reference.cc -- the TX8 reference standard (design doc §2): fixed
reference inputs, the 13 timed groups computed from the §2.2 workload table
with C = Fop/P + Eop/V and L = D/BW_class started together (group finishes at
max), and the §2.3 FFN accounting / capacity check.  No Excel time lookup
anywhere; every number is computed from the formulas below.
*/

#include "DieModel.hh"

#include <array>
#include <string>

namespace DieModel {

namespace {

// Fixed reference inputs (design doc §2.1).
constexpr double kB = 1.0;
constexpr double kSq = 1024.0;
constexpr double kSk = 1024.0;
constexpr double kH = 4096.0;
constexpr double kF = 14336.0;
constexpr double kHq = 32.0;
constexpr double kHkv = 8.0;
constexpr double kD = 128.0;

// Workload table (design doc §2.2).  A = B*Hq*Sq*Sk, J = Hkv*d, M = B*Sq.
struct RawGroup {
  const char* name;
  double fop;
  double eop;
  double d_bytes;
  const char* mem_class;  // "matrix" | "vector"
};

std::array<RawGroup, 13> build_raw_groups(double M, double J, double A) {
  return {{
      // RMSNorm: Eop 6*M*H, D 4*M*H+2*H (BF16 in/out + gamma, gamma once).
      {"RMSNorm_1", 0.0, 6.0 * M * kH, 4.0 * M * kH + 2.0 * kH, "vector"},
      // Q/O projection: Fop 2*M*H*H, D 2*(M*H + H*H + M*H).
      {"Q_proj", 2.0 * M * kH * kH, 0.0,
       2.0 * (M * kH + kH * kH + M * kH), "matrix"},
      // K/V projection: Fop 2*M*H*J, D 2*(M*H + H*J + M*J).
      {"K_proj", 2.0 * M * kH * J, 0.0, 2.0 * (M * kH + kH * J + M * J),
       "matrix"},
      {"V_proj", 2.0 * M * kH * J, 0.0, 2.0 * (M * kH + kH * J + M * J),
       "matrix"},
      // RoPE_QK: Eop 3*M*(H+J), D 4*M*(H+J)+4*Sq*d (cos/sin read once).
      {"RoPE_QK", 0.0, 3.0 * M * (kH + J),
       4.0 * M * (kH + J) + 4.0 * kSq * kD, "vector"},
      // QK^T: Fop 2*A*d, D 2*B*Hq*Sq*d + 2*B*Hkv*Sk*d + 4*A.
      {"QKT", 2.0 * A * kD, 0.0,
       2.0 * kB * kHq * kSq * kD + 2.0 * kB * kHkv * kSk * kD + 4.0 * A,
       "matrix"},
      // scale/mask/softmax: Eop 12*A, D 8*A.
      {"score_mul_softmax", 0.0, 12.0 * A, 8.0 * A, "vector"},
      // PV: Fop 2*A*d, Eop A (FP32->BF16 probability conversion per tile),
      // D 4*A + 2*B*Hkv*Sk*d + 2*B*Hq*Sq*d.
      {"AV/PV", 2.0 * A * kD, A,
       4.0 * A + 2.0 * kB * kHkv * kSk * kD + 2.0 * kB * kHq * kSq * kD,
       "matrix"},
      {"O_proj", 2.0 * M * kH * kH, 0.0,
       2.0 * (M * kH + kH * kH + M * kH), "matrix"},
      // residual add: Eop M*H, D 6*M*H.
      {"res_add_1", 0.0, M * kH, 6.0 * M * kH, "vector"},
      {"RMSNorm_2", 0.0, 6.0 * M * kH, 4.0 * M * kH + 2.0 * kH, "vector"},
      // fused FFN: Fop 6*M*H*F, Eop 16*M*F, D 6*H*F + 4*M*H (intermediate
      // activations stay in SRAM; no independent SiLU/Hadamard DRAM traffic).
      {"FFN", 6.0 * M * kH * kF, 16.0 * M * kF, 6.0 * kH * kF + 4.0 * M * kH,
       "matrix"},
      {"res_add_2", 0.0, M * kH, 6.0 * M * kH, "vector"},
  }};
}

}  // namespace

Tx8Result tx8_reference_block() {
  const double J = kHkv * kD;           // 1024
  const double M = kB * kSq;            // 1024
  const double A = kB * kHq * kSq * kSk;  // 33554432

  // Coefficients from the §2.1 formulas (DieTypes.hh, calibration namespace).
  const double P = calibration::matrix_peak_flop_s();
  const double V = calibration::vector_peak_eqop_s();
  const double bw_matrix = calibration::bw_matrix_bytes_s();
  const double bw_vector = calibration::bw_vector_bytes_s();

  Tx8Result r;
  const std::array<RawGroup, 13> raw = build_raw_groups(M, J, A);
  for (std::size_t i = 0; i < raw.size(); ++i) {
    Tx8GroupTiming& g = r.groups[i];
    g.name = raw[i].name;
    g.fop = raw[i].fop;
    g.eop = raw[i].eop;
    g.d_bytes = raw[i].d_bytes;
    g.mem_class = raw[i].mem_class;
    // Reference-executor semantics (design doc §2.2): the compute branch
    // PE(Fop) -> vec(Eop) and the memory stream start together; the group
    // completes at the later one.
    g.compute_s = g.fop / P + g.eop / V;
    g.mem_s = g.d_bytes / (raw[i].mem_class == std::string("vector")
                               ? bw_vector
                               : bw_matrix);
    g.total_s = g.compute_s > g.mem_s ? g.compute_s : g.mem_s;
  }

  // Subtotals (design doc §2.3): attention core = groups 2-9 (0-based 1..8),
  // block = groups 1-13.
  for (std::size_t i = 1; i <= 8; ++i) {
    r.attention_core_s += r.groups[i].total_s;
  }
  for (const Tx8GroupTiming& g : r.groups) {
    r.block_total_s += g.total_s;
  }

  // FFN intra-group accounting (design doc §2.3): compute-resource split of
  // the fused group.  gate/up/down each take one third of 6*M*H*F/P; SiLU is
  // 15*M*F/V and gate_mul 1*M*F/V (summing to the 16*M*F/V vector budget).
  r.ffn_gate_s = (6.0 * M * kH * kF / 3.0) / P;
  r.ffn_up_s = (6.0 * M * kH * kF / 3.0) / P;
  r.ffn_down_s = (6.0 * M * kH * kF / 3.0) / P;
  r.ffn_silu_s = 15.0 * M * kF / V;
  r.ffn_gate_mul_s = 1.0 * M * kF / V;

  // FFN reference traffic and external-memory service budget (§2.3): the
  // budget stays below the compute budget, so the group is compute-bound.
  r.ffn_traffic_bytes = 6.0 * kH * kF + 4.0 * M * kH;
  r.ffn_memory_budget_s = r.ffn_traffic_bytes / bw_matrix;

  // Reference capacity check with the workbook F_tile = 512 (design doc
  // §2.3): BF16 input M*H*2, FP32 output accumulator M*H*4, three weight
  // slabs 3*(2*H*F_tile) and three activations 3*(M*F_tile*2), against
  // 0.9 * 48e6 (90% of the 48 MB reference SRAM).
  constexpr double kFTile = 512.0;
  r.ffn_working_set_bytes =
      M * kH * 2.0 + M * kH * 4.0 + 3.0 * 2.0 * kH * kFTile +
      3.0 * M * kFTile * 2.0;
  r.sram_capacity_limit_bytes = 0.9 * 48e6;
  r.sram_capacity_ok = r.ffn_working_set_bytes < r.sram_capacity_limit_bytes;
  return r;
}

}  // namespace DieModel
