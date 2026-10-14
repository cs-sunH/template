/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
DieTypes.hh -- public types of the die-internal cost model library.

Sources of truth:
  - design doc (paper_write/die内建模设计指导/die内仿真器设计文档.md)
    sections 2 (TX8 reference standard) and 3 (target die hardware inputs);
  - cross-stage contract (/tmp/die_build/CONTRACTS.md) sections 1 (layout),
    2 (DieOpSpec schema), 3 (hardware configuration) and 4 (library API).

Stage A1 scope: DieModelError, the string-backed enums, the §2.1 calibration
constants, Hardware + core table, DieOpSpec with its strict parse declaration,
and the TX8 reference result types.  Stage A2 adds in this same header: the
§4.2 workload determination types, the immutable DiePlan stage DAG (contract
§5), CostResult and PlanCache.  Stage A3 adds the Execution action protocol
and the pauseable plan-advancement state machine (contract §4; implemented in
src/Execution.cc, C ABI in src/Capi.cc).
*/

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <json/json.hpp>

namespace DieModel {

// ---------------------------------------------------------------------------
// Error type (contract §1: namespace DieModel, DieModel::DieModelError)

class DieModelError : public std::runtime_error {
 public:
  explicit DieModelError(const std::string& what)
      : std::runtime_error(what) {}
};

// ---------------------------------------------------------------------------
// String-backed enums (contract §2 enum spellings; design doc §3.1)

enum class CoreType { Small, Large };
enum class Profile { MeshDieV1, Tx8ReferenceV1 };
enum class OpKind {
  Norm,
  QkvRope,
  Qk,
  ScaleMask,
  Softmax,
  Av,
  OProj,
  Residual,
  AttentionRegion,
  MlpUp,
  MlpActivation,
  MlpDown,
  FfnRegion,
  Logits,
  LocalRegion,
};
enum class Phase { Prefill, Decode };
enum class Dtype { Bf16, Fp32 };
enum class MlpVariant { Swiglu, Gelu };
enum class NormType { Rmsnorm, Layernorm };
enum class PositionEncoding { Rope, None };
enum class KvIdentity { Local, Staged, RemoteReady };
enum class FusionPolicy { Auto, None };

// to_string returns the exact JSON spelling; from_string only accepts that
// exact spelling (strict enum match, contract §2).
#define DIE_MODEL_ENUM_FN(Enum, ...)                                        \
  inline const char* to_string(Enum v) {                                    \
    static const char* kNames[] = {__VA_ARGS__};                            \
    return kNames[static_cast<std::size_t>(v)];                             \
  }                                                                         \
  inline bool from_string(const std::string& s, Enum& out) {                \
    static const char* kNames[] = {__VA_ARGS__};                            \
    for (std::size_t i = 0; i < sizeof(kNames) / sizeof(kNames[0]); ++i) {  \
      if (s == kNames[i]) {                                                 \
        out = static_cast<Enum>(i);                                         \
        return true;                                                        \
      }                                                                     \
    }                                                                       \
    return false;                                                           \
  }

DIE_MODEL_ENUM_FN(CoreType, "small", "large")
DIE_MODEL_ENUM_FN(Profile, "mesh_die_v1", "tx8_reference_v1")
DIE_MODEL_ENUM_FN(OpKind, "norm", "qkv_rope", "qk", "scale_mask", "softmax",
                  "av", "o_proj", "residual", "attention_region", "mlp_up",
                  "mlp_activation", "mlp_down", "ffn_region", "logits",
                  "local_region")
DIE_MODEL_ENUM_FN(Phase, "prefill", "decode")
DIE_MODEL_ENUM_FN(Dtype, "bf16", "fp32")
DIE_MODEL_ENUM_FN(MlpVariant, "swiglu", "gelu")
DIE_MODEL_ENUM_FN(NormType, "rmsnorm", "layernorm")
DIE_MODEL_ENUM_FN(PositionEncoding, "rope", "none")
DIE_MODEL_ENUM_FN(KvIdentity, "local", "staged", "remote_ready")
DIE_MODEL_ENUM_FN(FusionPolicy, "auto", "none")

#undef DIE_MODEL_ENUM_FN

// ---------------------------------------------------------------------------
// §2.1 calibration constants (design doc §2.1)

// The design doc requires these coefficients to be generated once from the
// formulas below and kept in double; no Excel time lookup is performed.
// configs/calibration_mesh_die_v1.json and configs/calibration_
// tx8_reference_v1.json mirror the same values, and configs_consistency_test
// enforces code == json (anti-drift).
namespace calibration {

// Calibration sample shape (design doc §2.1: B=1, Sq=Sk=1024, H=4096, F=14336).
inline constexpr double kSampleM = 1024.0;
inline constexpr double kSampleH = 4096.0;
inline constexpr double kSampleF = 14336.0;

// eta_compute = (6*M*H*F/128e12 + 16*M*F/1e12) / 0.003363 = 0.9079551923877489
// (design doc §2.1).  Shared by both profiles: mesh uses it as eta_pe/eta_vec,
// the TX8 reference uses it to build P and V.
inline double eta_compute() {
  return (6.0 * kSampleM * kSampleH * kSampleF / 128e12 +
          16.0 * kSampleM * kSampleF / 1e12) /
         0.003363;
}

// P = 128e12 * eta_compute FLOP/s (design doc §2.1).
inline double matrix_peak_flop_s() { return 128e12 * eta_compute(); }

// V = 1e12 * eta_compute eq-op/s (design doc §2.1).
inline double vector_peak_eqop_s() { return 1e12 * eta_compute(); }

// BW_matrix = 200e9 * 0.70 byte/s (design doc §2.1).
inline double bw_matrix_bytes_s() { return 200e9 * 0.70; }

// BW_vector = (4*M*H + 2*H) / 0.000246 byte/s; the numerator 16785408 bytes
// is one BF16 input + one BF16 output + gamma, gamma counted exactly once
// (design doc §2.1).
inline double bw_vector_bytes_s() {
  return (4.0 * kSampleM * kSampleH + 2.0 * kSampleH) / 0.000246;
}

// eta_memory_vector = BW_vector / 200e9 (design doc §2.1).
inline double eta_vector_mem() { return bw_vector_bytes_s() / 200e9; }

// eta_matrix_mem = 0.70 (BW_matrix = 200e9 * 0.70, design doc §2.1).
inline constexpr double kEtaMatrixMem = 0.70;

// eta_noc = 0.70 (design doc §3.2: T_NoC = max_edge(bytes / (0.70*link_peak))).
inline constexpr double kEtaNoc = 0.70;

// One launch per macro group, 1 us each, nine macro groups per op
// (design doc §6.1; mesh_die_v1 default).
inline constexpr double kLaunchUsPerMacroGroup = 1.0;

}  // namespace calibration

// ---------------------------------------------------------------------------
// Hardware (design doc §3.1; contract §3.1/§3.3)

// One row of the 14-row core table (design doc §3.1 table).
struct CoreTableEntry {
  double sram_mb_per_core = 0.0;
  CoreType core_type = CoreType::Small;
  double vec_g_eqop_per_core = 0.0;  // G eq-op/s per core
  double pe_gflops_per_core = 0.0;   // GFLOP/s per core
  double core_side_ref_um = 0.0;     // reference core edge length, um
};

// The full 14-row core table, embedded in the library (design doc §3.1).
// configs/core_table.json mirrors the same values; configs_consistency_test
// enforces code == json.
std::vector<CoreTableEntry> core_table();

// Per-die hardware description: the six §3.1 inputs, the matched core table
// row (cache), the derived resource scalars (contract §3.3 "derived") and the
// profile calibration coefficients.
struct Hardware {
  // --- six die inputs (contract §3.1 keys, values verbatim) ---
  double sram_mb_per_core = 0.0;
  CoreType core_type = CoreType::Large;
  double compute_area_mm2 = 0.0;
  std::uint64_t hbm_per_side = 0;
  std::uint64_t hbm_sides = 0;
  std::uint64_t noc_width_bits = 0;
  Profile profile = Profile::MeshDieV1;

  // --- matched core table row (equals one row of core_table()) ---
  CoreTableEntry core_row{};

  // --- derived (design doc §3.1 code block; frequency fixed at 1 GHz) ---
  double acore_mm2 = 0.0;  // Acore, mm^2
  std::uint64_t mesh_n = 0;  // n, mesh side length in cores
  std::uint64_t cores = 0;   // n*n
  double pe_peak_die = 0.0;  // FLOP/s
  double vec_peak_die = 0.0;  // eq-op/s
  double sram_usable_bytes_per_core = 0.0;  // 0.90 * sram_mb * 1e6
  double link_peak_bytes = 0.0;  // byte/s, one directed intra-die link
  std::uint64_t hbm_count = 0;
  double hbm_capacity_bytes = 0.0;
  double hbm_bandwidth_bytes = 0.0;  // byte/s

  // --- mesh profile coefficients (§6.1 defaults; values from §2.1 source) ---
  double eta_pe = 0.0;
  double eta_vec = 0.0;
  double eta_matrix_mem = 0.0;
  double eta_vector_mem = 0.0;
  double eta_noc = 0.0;
  double launch_us_per_macro_group = 0.0;

  // --- TX8 reference constants (design doc §2.1).  These are the fixed
  //     reference-standard numbers; they are independent of the die geometry
  //     above and are consumed by the tx8_reference_v1 profile path. ---
  double tx8_matrix_peak_flop_s = 0.0;
  double tx8_vector_peak_eqop_s = 0.0;
  double tx8_bw_matrix_bytes_s = 0.0;
  double tx8_bw_vector_bytes_s = 0.0;
  double tx8_extra_launch_s = 0.0;
  double tx8_extra_memory_latency_s = 0.0;
  bool full_score_materialization = false;
  bool skip_fully_masked_tiles = false;
};

// ---------------------------------------------------------------------------
// DieOpSpec (contract §2; design doc §4)

struct DieSpan {
  std::string request_id;
  std::uint64_t q_tokens = 0;     // >= 1
  std::uint64_t kv_tokens = 0;    // >= 1
  std::uint64_t q_start_pos = 0;  // absolute in-sequence query position
  std::uint64_t kv_start_pos = 0;  // absolute in-sequence KV start
};

struct DieModelDims {
  std::uint64_t hidden = 0;
  std::uint64_t ffn = 0;
  std::uint64_t vocab = 0;
  MlpVariant mlp_variant = MlpVariant::Swiglu;
  NormType norm_type = NormType::Rmsnorm;
  PositionEncoding position_encoding = PositionEncoding::Rope;
  std::uint64_t num_q_heads = 0;
  std::uint64_t num_kv_heads = 0;
  std::uint64_t head_dim = 0;
};

struct DieTpSlice {
  std::uint64_t tensor_parallel = 1;
  std::uint64_t tensor_parallel_rank = 0;
  std::uint64_t q_heads_local = 0;   // 0 is legal (empty shard, §4.2 rule 5)
  std::uint64_t kv_heads_local = 0;
  std::uint64_t ffn_local = 0;
  std::uint64_t vocab_local = 0;
};

struct DieKvAccess {
  KvIdentity identity = KvIdentity::Local;
};

struct DieOpSpec {
  std::uint64_t schema_version = 1;
  OpKind op_kind = OpKind::Norm;
  Phase phase = Phase::Prefill;
  Dtype dtype = Dtype::Bf16;
  DieModelDims model_dims{};
  DieTpSlice tp_slice{};
  std::vector<DieSpan> spans;
  std::uint64_t layer_count = 1;     // >= 1
  std::uint64_t weight_passes = 1;   // >= 1
  DieKvAccess kv_access{};
  FusionPolicy fusion_policy = FusionPolicy::Auto;
  std::string provenance;   // optional
  std::string config_hash;  // optional
};

// Strict parse of a die_op_spec JSON object (contract §2 schema): unknown
// keys rejected, exact types (unsigned integers must be unsigned, no
// floating point / negative / out-of-range values), exact enum spelling,
// spans >= 1, layer_count >= 1, weight_passes >= 1, and the basic shape
// relations of §4.2.  Throws DieModelError with field-path-prefixed
// messages on every violation.
std::shared_ptr<const DieOpSpec> parse_die_op_spec(const nlohmann::json& j);

// ---------------------------------------------------------------------------
// TX8 reference result (design doc §2.2/§2.3)

// One timed group of the reference block.  compute_s = Fop/P + Eop/V and
// mem_s = D/BW_class start together; total_s = max(compute_s, mem_s)
// (reference-executor semantics, design doc §2.2).
struct Tx8GroupTiming {
  const char* name = "";
  double fop = 0.0;            // matrix FLOP
  double eop = 0.0;            // vector eq-op
  double d_bytes = 0.0;        // actual memory traffic, bytes
  const char* mem_class = "";  // "matrix" | "vector" (§2.2 table)
  double compute_s = 0.0;
  double mem_s = 0.0;
  double total_s = 0.0;
};

struct Tx8Result {
  static constexpr std::size_t kGroupCount = 13;
  std::array<Tx8GroupTiming, kGroupCount> groups{};

  // Subtotals (design doc §2.3): groups 2-9 and groups 1-13.
  double attention_core_s = 0.0;
  double block_total_s = 0.0;

  // FFN intra-group accounting (design doc §2.3): a compute-side split of the
  // FFN group, not five extra kernels.  gate/up/down split 6*M*H*F/3 each;
  // SiLU = 15*M*F/V and gate_mul = M*F/V (sum = the group's compute budget).
  double ffn_gate_s = 0.0;
  double ffn_up_s = 0.0;
  double ffn_down_s = 0.0;
  double ffn_silu_s = 0.0;
  double ffn_gate_mul_s = 0.0;

  // FFN reference traffic 6*H*F + 4*M*H and its external-memory service
  // budget traffic/BW_matrix (design doc §2.3).
  double ffn_traffic_bytes = 0.0;
  double ffn_memory_budget_s = 0.0;

  // Reference capacity check with the workbook F_tile = 512 (design doc
  // §2.3): input, FP32 output accumulator, three weight slabs and three
  // activations vs 0.9 * 48e6.  A single working-set check; it does not
  // prove full double buffering.
  double ffn_working_set_bytes = 0.0;
  double sram_capacity_limit_bytes = 0.0;
  bool sram_capacity_ok = false;
};

// ---------------------------------------------------------------------------
// Internal strict-JSON helpers (field-path-prefixed errors).  Shared by
// Workload.cc (die_op_spec) and Hardware.cc (die node).  Not part of the
// A2-facing public API.

namespace detail {

[[noreturn]] void fail(const std::string& path, const std::string& message);

// Rejects any key of obj not listed in allowed; the message names the key.
void reject_unknown_keys(const nlohmann::json& obj,
                         const std::vector<const char*>& allowed,
                         const std::string& path);

// Returns obj[key]; throws if missing.
const nlohmann::json& member(const nlohmann::json& obj, const char* key,
                             const std::string& path);

// Strict unsigned integer: is_number_unsigned() only (floats, signed values
// and strings are rejected).
std::uint64_t require_uint(const nlohmann::json& obj, const char* key,
                           const std::string& path);

// Any JSON number, returned as double.
double require_number(const nlohmann::json& obj, const char* key,
                      const std::string& path);

std::string require_string(const nlohmann::json& obj, const char* key,
                           const std::string& path);

// Exact string-enum match; the error lists the valid spellings.
std::string require_enum_string(const nlohmann::json& obj, const char* key,
                                const std::vector<const char*>& valid,
                                const std::string& path);

}  // namespace detail

// ---------------------------------------------------------------------------
// Workload determination (design doc §4.2) -- stage A2
//
// derive_workload() turns one DieOpSpec into the mathematical work of that
// op (matrix FLOP / vector eq-ops / byte skeleton) WITHOUT any tiling or
// mapping decision.  Mapping (src/Mapping.cc) turns these quantities into
// tile-level plans; the plan totals may legitimately exceed these floors
// (tile rectangles crossing the causal diagonal, reloads, partial sums).

// One constituent GEMM of a linear op, logical [m,k]x[k,n] (design doc §4.2).
struct DieGemmShape {
  std::uint64_t m = 0;
  std::uint64_t k = 0;
  std::uint64_t n = 0;
};

// Attention summary of one span (design doc §4.2 rules 2 and 7): attention is
// charged per span as sum(q_i*k_i) with that span's own kv range and absolute
// causal positions, never sum(q)*sum(k).
struct DieAttentionSpanWork {
  std::uint64_t q_tokens = 0;
  std::uint64_t kv_tokens = 0;
  std::uint64_t q_start_pos = 0;
  std::uint64_t kv_start_pos = 0;
  std::uint64_t valid_pairs = 0;      // causally valid q*k pairs
  std::uint64_t rectangle_pairs = 0;  // q_tokens*kv_tokens (materialization)
};

struct WorkloadQuantities {
  OpKind op_kind = OpKind::Norm;
  Dtype dtype = Dtype::Bf16;
  std::uint64_t elem_bytes = 2;   // activation/weight element size in bytes

  // Linear dimension M = sum(spans.q_tokens) -- same physical batch merged
  // for all linear ops (design doc §4.2 rule 1); decode M=1 is never padded
  // (rule 7).
  std::uint64_t m_tokens = 0;

  std::uint64_t layer_count = 1;
  std::uint64_t weight_passes = 1;
  // Macro-group launch instances = layer_count * weight_passes (design doc
  // §6.1; frozen in the stage A2 report).
  std::uint64_t launch_instances = 1;

  // Per-layer math work (one physical execution of the op's math).
  double flops_per_layer = 0.0;
  double eqops_per_layer = 0.0;

  // Per-layer byte skeleton, floor (untiled: read once / written once /
  // KV once); mapping produces the actual tiled bytes.
  double weight_bytes_per_pass = 0.0;  // per layer AND per weight pass
  double act_in_bytes_per_layer = 0.0;
  double act_out_bytes_per_layer = 0.0;
  double kv_bytes_per_layer = 0.0;     // K/V reads (attention family)
  double aux_bytes_per_layer = 0.0;    // cos/sin tables etc (vector class)

  // Memory-class split of the per-layer floor bytes (design doc §2.2 class
  // column / §6.1 t_mem_alone).
  double matrix_bytes_per_layer = 0.0;
  double vector_bytes_per_layer = 0.0;

  // Totals (per-layer quantities x layer_count; weight bytes additionally
  // x weight_passes).
  double total_flops = 0.0;
  double total_eqops = 0.0;
  double total_weight_bytes = 0.0;
  double total_matrix_bytes = 0.0;
  double total_vector_bytes = 0.0;

  // Skeleton details consumed by the mapper.
  std::vector<DieGemmShape> gemms;                 // merged-M linear ops
  std::vector<DieAttentionSpanWork> attention_spans;  // per span
  std::uint64_t hidden = 0;
  std::uint64_t ffn_local = 0;
  std::uint64_t vocab_local = 0;
  std::uint64_t q_heads_local = 0;
  std::uint64_t kv_heads_local = 0;
  std::uint64_t head_dim = 0;
  MlpVariant mlp_variant = MlpVariant::Swiglu;
  NormType norm_type = NormType::Rmsnorm;
  PositionEncoding position_encoding = PositionEncoding::Rope;
  // q_heads_local * sum(valid_pairs): the per-head attention pair floor.
  double attention_pairs = 0.0;
};

// Throws DieModelError on structurally inconsistent input (e.g. attention
// spans without head information); pure arithmetic otherwise.
WorkloadQuantities derive_workload(const DieOpSpec& spec);

// ---------------------------------------------------------------------------
// Plan representation (contract §5; structure frozen in the stage A2 report)
//
// A DiePlan is an immutable, cheaply shared description of how one DieOpSpec
// executes alone on one Hardware: a mapping summary plus a repeat-compressed
// stage DAG.  Stage A3's Execution state machine and estimate_alone() both
// consume exactly this representation.

// Resource class of a stage.  PE/VEC/NOC/MEM/MEM_NOC are the five contract
// §5 classes; TIMER carries the §6.1 launch overhead (and any other internal
// timed section that occupies no resource slot), which the contract's
// Execution protocol returns as kind=Timer.
enum class StageResource { PE, VEC, NOC, MEM, MEM_NOC, TIMER };

// Dependency of one stage on another.  Instance j of a stage depends on
// instance (j - lag) of the predecessor stage; negative indices are vacuous;
// DieStageDep::kAllDep means "all instances of the predecessor".
struct DieStageDep {
  std::uint32_t stage = 0;
  std::uint32_t lag = 0;
  static constexpr std::uint32_t kAllDep = 0xFFFFFFFFu;
  DieStageDep() = default;
  DieStageDep(std::size_t s, std::uint32_t l)
      : stage(static_cast<std::uint32_t>(s)), lag(l) {}
};

// Memory request triple of contract §5/§6.2: actual bytes, service units and
// the class eta.  service_units already includes the HBM-granularity rule
// max(sum(U_j), N_hbm*max(U_j)) with U_j = D_j/eta (design doc §6.2).
struct DieMemoryRequest {
  double actual_bytes = 0.0;
  double service_units = 0.0;
  double eta = 0.0;
  bool matrix_class = true;
};

// One repeat-compressed stage.  All work quantities are PER INSTANCE (one
// wave); wave_repeat instances run in index order on the stage's slot.
struct DieStage {
  std::string label;
  StageResource resource = StageResource::PE;
  std::uint64_t macro_group = 0;   // op_kind -> macro group table (A2 report)
  std::uint64_t wave_repeat = 1;

  double pe_core_flop = 0.0;   // max-core FLOP per instance (PE)
  double vec_core_eqop = 0.0;  // max-core eq-op per instance (VEC)
  double noc_edge_bytes = 0.0;  // max directed-edge bytes per instance
  double timer_seconds = 0.0;   // per instance (TIMER)
  DieMemoryRequest mem;         // per instance (MEM / MEM_NOC)

  std::vector<DieStageDep> deps;
};

// Mapping summary of a plan (contract §5 "tile/核心组/上下文数/循环类型/
// 广播决策").  Tile/loop spelling is frozen in the stage A2 report.
struct DiePlanSummary {
  OpKind op_kind = OpKind::Norm;
  Profile profile = Profile::MeshDieV1;
  std::string loop_kind;  // frozen spellings, see stage A2 report
  std::uint64_t m_tile = 0;
  std::uint64_t n_tile = 0;
  std::uint64_t k_tile = 0;
  std::uint64_t group_rows = 1;
  std::uint64_t group_cols = 1;
  std::uint64_t contexts = 1;
  std::uint64_t k_parallel = 1;
  bool weight_resident = false;   // GEMM loop type (design doc §5.2)
  bool double_buffered = true;    // S=4mk+4kn+4mn vs 2mk+2kn+4mn
  bool fused = false;             // region fusion in use (§5.4/§5.5)
  bool fallback = false;          // fused region fell back to separate ops
  bool streaming_two_pass = false;  // vector op could not keep a row (§5.3)
  double sram_peak_per_core = 0.0;
  std::uint64_t layer_count = 1;
  std::uint64_t weight_passes = 1;
  std::uint64_t launch_instances = 1;
  std::uint64_t waves = 0;           // per-layer wave count (repeat/L)
  double weight_hbm_bytes = 0.0;     // actual HBM weight traffic of the op
};

// Assembly form of a plan; DiePlan freezes it behind a const pointer.
struct DiePlanData {
  std::shared_ptr<const DieOpSpec> spec;
  Hardware hardware;
  DiePlanSummary summary;
  std::vector<DieStage> stages;
  double flops = 0.0;
  double eqops = 0.0;
  double matrix_bytes = 0.0;
  double vector_bytes = 0.0;
  double matrix_service_units = 0.0;
  double vector_service_units = 0.0;
  double noc_total_link_bytes = 0.0;  // sum over directed edges of all bytes
  double noc_max_edge_bytes = 0.0;    // busiest single directed edge
};

// Immutable plan: cheap to copy (shared const data), no mutators.
class DiePlan {
 public:
  DiePlan() = default;
  explicit DiePlan(std::shared_ptr<const DiePlanData> data);

  bool valid() const { return data_ != nullptr; }
  const DieOpSpec& spec() const;
  const Hardware& hardware() const;
  const DiePlanSummary& summary() const;
  const std::vector<DieStage>& stages() const;
  double flops() const;
  double eqops() const;
  double matrix_bytes() const;
  double vector_bytes() const;
  double matrix_service_units() const;
  double vector_service_units() const;
  double noc_total_link_bytes() const;
  double noc_max_edge_bytes() const;

 private:
  std::shared_ptr<const DiePlanData> data_;
};

// ---------------------------------------------------------------------------
// Isolated-run estimate (contract §4 field set)

struct CostResult {
  Profile profile = Profile::MeshDieV1;
  double total_seconds = 0.0;
  // Resource-slot busy times of the isolated run (die-level conservative
  // serialization, design doc §6.1).  A MEM_NOC stage occupies both the MEM
  // and NOC slots for its whole max(t_mem, t_NoC) duration.
  double pe_seconds = 0.0;
  double vec_seconds = 0.0;
  double noc_seconds = 0.0;
  double mem_seconds = 0.0;
  double launch_seconds = 0.0;
  std::uint64_t launch_instances = 0;

  double flops = 0.0;        // actual executed matrix FLOP (sum over cores)
  double eqops = 0.0;        // actual executed vector eq-ops
  double matrix_actual_bytes = 0.0;
  double vector_actual_bytes = 0.0;
  double matrix_service_units = 0.0;
  double vector_service_units = 0.0;
  double noc_link_bytes_total = 0.0;  // sum over directed edges
  double noc_edge_bytes_max = 0.0;    // busiest single directed edge
  double sram_peak_per_core = 0.0;

  bool fused = false;
  bool fallback = false;

  // ns integer suggestion for event-clock reservation (design doc §6.4):
  // max(1, ceil(total_seconds*1e9)) for positive totals, else 0.
  std::uint64_t total_ns_suggested = 0;
};

// LRU plan cache (contract §5: at most 4096 plans per model instance; the
// library itself is not thread-sensitive).  compile() stays pure; A3's model
// object and stage B own a PlanCache and key it with plan_cache_key().
class PlanCache {
 public:
  explicit PlanCache(std::size_t capacity = 4096);
  std::shared_ptr<const DiePlan> get(const std::string& key) const;
  void put(const std::string& key, std::shared_ptr<const DiePlan> plan);
  std::size_t size() const;
  std::size_t capacity() const;

 private:
  std::size_t capacity_;
  mutable std::list<std::pair<std::string, std::shared_ptr<const DiePlan>>>
      lru_;
  mutable std::unordered_map<
      std::string,
      std::list<std::pair<std::string,
                          std::shared_ptr<const DiePlan>>>::iterator>
      index_;
};

// Frozen cache-key hash (stage A2 report): FNV-1a 64 over an explicitly
// ordered canonical string of hardware summary + coefficient version tag +
// every spec field that can change the plan (incl. dtype/spans/layer/TP/KV
// ranges/fusion flags/config_hash).  16 lowercase hex digits.
std::string plan_cache_key(const DieOpSpec& spec, const Hardware& hw);

// ---------------------------------------------------------------------------
// Execution: externally driven plan advancement (contract §4; design doc
// §6.3/§6.4/§8.1) -- stage A3.
//
// An Execution is the "externally driven version" of the isolated list
// schedule behind estimate_alone() (detail::evaluate_stages in CostModel.cc):
// the SAME Kahn ready order ((stage<<32)|instance), the SAME conservative
// serialization of same-class stages on the PE/VEC/NOC/MEM slots and the SAME
// DieStageDep{stage,lag} semantics.  The difference is who provides time and
// memory completion:
//
//   - internal stages (TIMER/PE/VEC/NOC and the NoC leg of MEM_NOC) are timed
//     by the machine itself; next_action() reports the distance to the next
//     internal event as kind=Timer and the driver advances the clock;
//   - MEM/MEM_NOC stages are handed to the external memory port as
//     kind=MemoryRequest (at most ONE outstanding request per execution; the
//     next request is only issued after complete_memory() for the current
//     one); the driver services it at whatever rate the real arbiter
//     provides and calls complete_memory() once the clock is at (or past)
//     the completion moment;
//   - a MEM_NOC stage holds BOTH the MEM and the NOC slot from its start
//     until the later of (external memory completion, internal NoC leg end).
//
// The machine holds no network or KV state and performs no hidden caching:
// it is mutable per-run state, constructed fresh from an immutable
// shared_ptr<const DiePlan> for every outer node (design doc §8.5 "执行句柄
// 不入缓存").

// One step of the action protocol (contract §4).  Field use per kind:
//   Timer         ns = nanoseconds to the next internal event (>= 1);
//   MemoryRequest stage_id/actual_bytes/service_units = DieStage::mem triple
//                 of the instance whose service now starts; ns = 0;
//   Done          ns = accumulated elapsed_ns of the whole execution.
struct ExecutionAction {
  enum class Kind { Timer, MemoryRequest, Done };
  Kind kind = Kind::Timer;
  std::uint64_t ns = 0;  // Timer: delta; Done: elapsed_ns
  std::uint64_t stage_id = 0;
  double actual_bytes = 0.0;
  double service_units = 0.0;
};

// Auditable view of the current internal service frontier. This is derived
// from the immutable plan plus the live cursor; it does not advance work.
enum class InternalSliceClass {
  Idle,
  PurePe,
  PureVec,
  InternalNoc,
  MemoryLatency,
  MemoryService,
  MemNocResidual,
  MixedFrontier,
  Unknown,
};

struct InternalStageSlice {
  std::uint32_t stage_id = 0;
  std::uint64_t wave_index = 0;
  StageResource resource = StageResource::TIMER;
  std::uint64_t start_ns = 0;
  std::uint64_t end_ns = 0;
  std::uint64_t original_service_ns = 0;
  std::uint64_t remaining_service_ns = 0;
  double original_work = 0.0;  // max-core FLOP for PE; eq-op for VEC
  double remaining_work = 0.0;
  double memory_actual_bytes = 0.0;
  double memory_service_units = 0.0;
  bool memory_pending = false;
  std::uint64_t hardware_min_ns = 0;
};

struct InternalActionToken {
  std::string template_key;
  std::uint64_t flat_instance_id = 0;
  std::uint64_t frontier_generation = 0;
  bool valid = false;
};

struct InternalSliceInfo {
  InternalActionToken action_token;  // plan + instance + frontier generation
  std::string template_key;
  InternalSliceClass classification = InternalSliceClass::Unknown;
  std::uint64_t now_ns = 0;
  std::uint64_t next_boundary_ns = 0;
  bool has_next_boundary = false;
  std::vector<InternalStageSlice> active_stages;
  bool partial_service_supported = false;
  std::string reason;
};

struct InternalTemplateSliceEntry {
  std::uint32_t stage_id = 0;
  std::uint64_t wave_index = 0;
  StageResource resource = StageResource::TIMER;
  double original_work = 0.0;
  std::uint64_t original_service_ns = 0;
  std::uint64_t hardware_min_ns = 0;
};

/// Stable authority to consume a bounded, already-proven consecutive
/// same-stage PE/VEC timer sequence. The first action token anchors the plan;
/// later wave tokens are derived and checked in order as each boundary is
/// crossed. Callers may shorten original_service_ticks to commit a prefix,
/// but may not extend it or change its values.
struct InternalServiceWindowToken {
  InternalActionToken first_action;
  std::uint32_t stage_id = 0;
  std::uint64_t first_wave_index = 0;
  std::vector<std::uint64_t> original_service_ticks;
  std::uint64_t service_already_applied = 0;
  std::uint64_t completed_slices = 0;
  bool valid = false;
};

/// Bounded read-only lookahead for one contiguous homogeneous timer window.
struct InternalTimerWindowInfo {
  bool supported = false;
  std::string reason;
  InternalServiceWindowToken token;
  std::vector<InternalTemplateSliceEntry> slices;
};

// ---------------------------------------------------------------------------
// v12 Phase 4 (方案 §6.2): process-level STRUCTURAL NEGATIVE cache for
// query_internal_timer_window.
//
// Unlike the v11 epoch-keyed memo (same layer as the queried state), this
// cache is keyed ONLY by (plan cache key, window-approximation config
// fingerprint) and therefore crosses Execution instances and run states.
// That is only Sound for a rejection whose COMPLETE return value -- the
// reason string included -- is provably identical in EVERY reachable state
// of EVERY Execution of the plan. The Phase 4 audit
// (docs/PHASE4_K2_CACHE_DESIGN.md) established:
//
//   1. The window growth ring (Execution.cc, "safe_count" loop) can never
//      turn supported=false: every ring break only TRUNCATES the window to
//      >=1 slice and the query still returns supported=true. Structural
//      growth-ring predicates therefore prove nothing about acceptance.
//   2. Every other rejection reason is a function of the live scheduler
//      state (finished / mem_outstanding / frontier composition /
//      next_mem_issue_at, the last one externally driven through
//      complete_memory), except for the class of plans that are finished
//      AT CONSTRUCTION: total==0, or every instance is a zero-duration
//      internal stage with no MEM/MEM_NOC instance (the §6.4 synchronous
//      zero-time rule drains the whole DAG inside the constructor's run()).
//      For exactly those plans the reachable state set is the singleton
//      {finished} and the query provably always returns
//      {supported=false, reason="execution-finished"}.
//
// P_struct(plan) below names that class; Execution computes it once from the
// immutable flattened arrays and gates BOTH the probe and the write below,
// so a hit can never change an accept/reject result or its reason
// (supported=true is never stored). The production fire-face is effectively
// empty (production plans always carry positive launch/compute/memory
// durations); the delivered value is the audited, fingerprint-keyed,
// Sound-by-construction wiring plus the test hooks.

namespace k2_window_approx_config {

// Compile-time REGISTRATION of the current production window-approximation
// configuration (v12 §11.1 frozen values: tile limit 2 ("K2"), local error
// 0 ppm ("ppm0"), refine margin 20000 ppm ("margin20000")). The runtime
// authority for these values is PipelineKernelContract on the astra-sim side
// (ExecutionDriven/PipelineDescriptor.hh); die_model is the LOWER layer of
// that dependency and cannot reach it, so the current production values are
// registered here as a compile-time table. A future arm that changes the
// production configuration MUST either update this table or inject a
// distinct fingerprint through
// K2TimerWindowNegativeCache::set_config_fingerprint (entries under
// different fingerprints are mutually invisible; exact and approx modes can
// never share a conclusion).
inline constexpr std::uint32_t kFingerprintSchemaVersion = 1;
inline constexpr std::uint32_t kWindowTileLimit = 2;
inline constexpr std::uint32_t kWindowLocalErrorPpm = 0;
inline constexpr std::uint32_t kWindowRefineMarginPpm = 20000;

inline std::string production_fingerprint() {
  return "k2win-neg-schema" + std::to_string(kFingerprintSchemaVersion) +
         "|tile_limit=" + std::to_string(kWindowTileLimit) +
         "|local_error_ppm=" + std::to_string(kWindowLocalErrorPpm) +
         "|refine_margin_ppm=" + std::to_string(kWindowRefineMarginPpm);
}

}  // namespace k2_window_approx_config

// LRU negative-conclusion cache (the PlanCache caliber: single-driver
// process, no locking; entries are values copied in and out). The config
// fingerprint is part of every entry key, so switching fingerprints never
// serves a foreign-configuration conclusion and old entries stay available
// when the fingerprint switches back.
class K2TimerWindowNegativeCache {
 public:
  explicit K2TimerWindowNegativeCache(
      std::size_t capacity = 4096,
      std::string config_fingerprint =
          k2_window_approx_config::production_fingerprint());

  // Extension point / test hook (see k2_window_approx_config above).
  void set_config_fingerprint(std::string fingerprint);
  const std::string& config_fingerprint() const { return fingerprint_; }

  // Master switch for the whole negative-cache path (Execution consults it
  // only when enabled). Default: enabled.
  void set_enabled(bool enabled) { enabled_ = enabled; }
  bool enabled() const { return enabled_; }

  void clear();
  std::size_t size() const;  // total entries across all fingerprints used

  // Probe/insert under the CURRENT config fingerprint; find() promotes the
  // entry to the LRU front like PlanCache::get. Execution only ever inserts
  // rejections of P_struct plans (see the audit note above).
  std::optional<InternalTimerWindowInfo> find(
      const std::string& plan_cache_key) const;
  void insert(const std::string& plan_cache_key,
              InternalTimerWindowInfo rejection);

 private:
  std::string entry_key(const std::string& plan_cache_key) const;

  std::size_t capacity_;
  bool enabled_ = true;
  std::string fingerprint_;
  mutable std::list<std::pair<std::string, InternalTimerWindowInfo>> lru_;
  mutable std::unordered_map<
      std::string,
      std::list<std::pair<std::string, InternalTimerWindowInfo>>::iterator>
      index_;
};

// The process-level instance every Execution consults (function-local
// static: initialized on first use; the library's single-driver contract
// makes an unsynchronized singleton the same caliber as PlanCache).
K2TimerWindowNegativeCache& k2_timer_window_negative_cache();

struct ExecutionProgress {
  struct StageCount {
    std::uint32_t stage_id = 0;
    StageResource resource = StageResource::TIMER;
    std::uint64_t completed = 0;
    std::uint64_t total = 0;
    double remaining_work = 0.0;
    double remaining_noc_bytes = 0.0;
    double remaining_actual_bytes = 0.0;
    double remaining_service_units = 0.0;
    std::uint64_t remaining_internal_service_ns = 0;
  };
  std::uint64_t now_ns = 0;
  std::uint64_t elapsed_ns = 0;
  bool finished = false;
  bool memory_pending = false;
  std::uint64_t memory_stage_id = 0;
  bool has_memory_instance_id = false;
  std::uint32_t memory_instance_id = 0;
  double pending_memory_actual_bytes = 0.0;
  double pending_memory_service_units = 0.0;
  std::uint64_t external_memory_wait_elapsed_ns = 0;
  // Deterministic accounting for completion-batch cleanup of the live active
  // frontier. These counters describe actual vector compaction work; they do
  // not affect scheduling or the externally driven action protocol.
  std::uint64_t active_id_compaction_calls = 0;
  std::uint64_t active_id_entries_examined = 0;
  std::uint64_t active_id_entries_moved = 0;
  std::uint64_t active_id_entries_removed = 0;
  InternalSliceInfo frontier;
  std::vector<StageCount> stages;
};

/// Exact immutable preview of the current consecutive internal Timer stretch.
/// The preview stops immediately before reporting a MemoryRequest and does
/// not advance the authoritative Execution cursor.
struct TimerStretchPreview {
  std::uint64_t external_delay_ns = 0;
  std::uint64_t timer_action_count = 0;
  bool reaches_memory_request = false;
  bool reaches_done = false;
};

/// Always-on preview attribution counters (integer diagnostics only; they
/// never affect scheduling or the action protocol). E/T keep the v8 caliber;
/// U is unified across every exit path (including exceptions). Since the v9
/// transactional restructure the walk takes no frontier copy at all, so the
/// copy measures F/copy_bytes are structurally zero and the entry frontier
/// sizes are reported separately as the four S_* totals/maxima.
struct PreviewCounters {
  std::uint64_t calls = 0;             // E: entry attempts
  std::uint64_t walk_timers = 0;       // T: walked timer boundaries
  std::uint64_t undo_entries = 0;      // U: dense undo records (all exits)
  std::uint64_t frontier_entries = 0;  // F: copied frontier entries (0)
  std::uint64_t j_entries = 0;         // J: heap/events undo records
  std::uint64_t s_heap_total = 0;
  std::uint64_t s_events_total = 0;
  std::uint64_t s_future_starts_total = 0;
  std::uint64_t s_active_ids_total = 0;
  std::uint64_t s_heap_max = 0;
  std::uint64_t s_events_max = 0;
  std::uint64_t s_future_starts_max = 0;
  std::uint64_t s_active_ids_max = 0;
  std::uint64_t copy_bytes = 0;           // frontier snapshot bytes (0)
  std::uint64_t active_scan_entries = 0;  // active-id scans in preview (0)
  std::uint64_t undo_peak = 0;
  std::uint64_t journal_peak = 0;
  std::uint64_t exit_memory = 0;
  std::uint64_t exit_done = 0;
  std::uint64_t exit_exception = 0;
  std::uint64_t heap_growth_events = 0;
  std::uint64_t heap_growth_moves = 0;
  std::uint64_t events_growth_events = 0;
  std::uint64_t events_growth_moves = 0;
};

class Execution {
 public:
  // Throws DieModelError on a null/empty plan or a cyclic stage DAG.
  explicit Execution(std::shared_ptr<const DiePlan> plan);
  ~Execution();

  // Non-copyable mutable run state (create one Execution per run).
  Execution(const Execution&) = delete;
  Execution& operator=(const Execution&) = delete;

  // Next action of the protocol.  While idle and unfinished it always makes
  // progress; repeated calls without advance_time()/complete_memory() return
  // the same action (idempotent).  A MemoryRequest returned again before its
  // complete_memory() is a re-report of the SAME outstanding request, not a
  // new one.
  ExecutionAction next_action();

  /// Read-only equivalent of the legacy contiguous-Timer coalescing walk.
  /// It runs a journal-based transaction on the authoritative scheduler
  /// state: heap/events writes are undone in reverse, the dense per-run
  /// arrays roll back through the undo log, the scalar cursor is restored
  /// from an entry snapshot, and the derived frontiers (future_starts /
  /// active_ids) are not maintained during the walk because no scheduling
  /// decision or return-value component reads them. The caller still
  /// advances the real machine when the returned delay expires.
  TimerStretchPreview preview_consecutive_internal_timers() const;

  // Advances the internal clock by ns (delta, uint64 ns) and completes every
  // internal stage whose end time is <= the new clock.  No-op once finished.
  // Throws DieModelError on clock overflow.
  void advance_time(std::uint64_t ns);

  // External memory-port completion callback.  The driver must have advanced
  // the clock to (or past) the completion moment before calling.  Throws
  // DieModelError if no request is outstanding or stage_id does not match it.
  void complete_memory(std::uint64_t stage_id);

  /// Return the outstanding external memory stage in O(1), without exporting
  /// the full execution progress snapshot.
  std::optional<std::uint32_t> pending_memory_stage_id() const;

  /// Read the current internal stage frontier without advancing the cursor.
  InternalSliceInfo describe_internal_slice() const;

  /// Query a bounded wave slice from the immutable plan. No execution or
  /// future events are materialized; callers should keep count small.
  std::vector<InternalTemplateSliceEntry> query_template_slice(
      std::uint32_t stage_id, std::uint64_t position,
      std::uint64_t count) const;

  /// Read-only proof of up to max_count consecutive single-stage PE/VEC timer
  /// actions. It does not clone or replay the plan and stops at any mixed,
  /// memory, NoC, or unknown boundary.
  InternalTimerWindowInfo query_internal_timer_window(
      std::uint32_t max_count) const;

  /// Apply original internal service against one stable PE/VEC action
  /// token. Stale tokens, mixed frontiers, and crossing the current event
  /// boundary are rejected.
  void advance_internal_service(const InternalActionToken& action_token,
                                std::uint64_t delta_original_ns);

  /// Apply a bounded original-service prefix against a stable homogeneous
  /// window token, crossing only the exact sequential waves named by that
  /// token. A stale token or an amount past its window fails before mutation.
  void advance_internal_service_window(InternalServiceWindowToken& token,
                                       std::uint64_t delta_original_ns);

  /// Advance global time only while waiting on the named external memory
  /// request; completion still requires complete_memory(stage_id).
  void advance_external_wait(std::uint64_t delta_global_ns,
                             std::uint64_t expected_memory_stage);

  /// Compact immutable snapshot of the live logical cursor.
  ExecutionProgress export_progress() const;

  bool finished() const;
  // Max completion time over the stages completed so far (the makespan once
  // finished()).
  std::uint64_t elapsed_ns() const;

  /// v8/v9 preview attribution counters (always-on diagnostics, the
  /// active_id_* caliber): E calls, T walked timer boundaries, U journaled
  /// dense writes (v9: unified across all exits), F frontier entries copied
  /// at entry (structurally zero since v9).
  std::uint64_t preview_calls() const;
  std::uint64_t preview_walk_timers() const;
  std::uint64_t preview_undo_entries() const;
  std::uint64_t preview_frontier_entries() const;

  /// Full v9 preview attribution snapshot (PreviewCounters): E/T/U/F plus
  /// J, the four per-container S totals/maxima, undo/journal peaks, exit
  /// categories and transaction capacity-growth counts.
  PreviewCounters preview_counters() const;

 private:
  friend struct ExecutionTestAccess;
  std::uint64_t template_key_computations_for_test() const;
  // v11 Phase 4 (K2 memo): hit/miss counters of the epoch-keyed
  // query_internal_timer_window memo, exposed only through the test-access
  // friend so fixtures can prove the memo actually engaged (a twin-execution
  // equality check would pass vacuously with the memo silently disabled).
  std::uint64_t timer_window_memo_hits_for_test() const;
  std::uint64_t timer_window_memo_misses_for_test() const;

  // v12 Phase 4 (K2 structural negative cache, 方案 §6.2): engagement
  // counters and the P_struct verdict of this Execution's plan, exposed only
  // through the test-access friend so fixtures can prove the negative cache
  // actually fired (and, for realistic plans, actually did NOT).
  std::uint64_t negative_cache_hits_for_test() const;
  std::uint64_t negative_cache_writes_for_test() const;
  bool structural_negative_for_test() const;

  struct Impl;
  // Held by value to keep the header free of scheduling internals; defined
  // in src/Execution.cc.
  std::unique_ptr<Impl> impl_;
};

// ---------------------------------------------------------------------------
// Internal mapping interface (src/Mapping.cc -> src/CostModel.cc).  Not part
// of the public API; declared here so the two translation units share types.

namespace detail {

// Isolated-run list schedule of a stage list (implemented in CostModel.cc,
// used by both compile() candidate selection and estimate_alone; the same
// die-level conservative model, design doc §6.1).
struct ScheduleResult {
  double total_seconds = 0.0;
  double pe_seconds = 0.0;
  double vec_seconds = 0.0;
  double noc_seconds = 0.0;
  double mem_seconds = 0.0;
};

ScheduleResult evaluate_stages(const std::vector<DieStage>& stages,
                               const Hardware& hw);

// Per-instance duration of one stage in SECONDS under the same §6.1 model
// (defined in CostModel.cc next to evaluate_stages; shared by candidate
// pruning, evaluate_stages and the stage A3 Execution machine so there is
// exactly one duration truth).  For MEM/MEM_NOC this is the full isolated
// duration (the memory leg at full bandwidth); TIMER returns timer_seconds.
double stage_instance_seconds(const DieStage& s, const Hardware& hw);

// The NoC leg alone of a MEM_NOC instance in SECONDS (0 for NOC-free
// profiles, e.g. tx8_reference_v1).  The Execution machine times this leg
// internally while the memory leg is served externally.
double stage_noc_leg_seconds(const DieStage& s, const Hardware& hw);

// One fully specified mapping candidate with its stage DAG and exact totals.
struct Candidate {
  DiePlanSummary summary;
  std::vector<DieStage> stages;
  double flops = 0.0;
  double eqops = 0.0;
  double matrix_bytes = 0.0;
  double vector_bytes = 0.0;
  double matrix_service_units = 0.0;
  double vector_service_units = 0.0;
  double noc_total_link_bytes = 0.0;
  double noc_max_edge_bytes = 0.0;
};

// Optional enumeration diagnostics used by the focused mapping unit test.
// Production callers leave this null; these counts do not participate in a
// candidate, its ranking, or any physical quantity.
struct EnumerationDiagnostics {
  std::uint64_t attention_aggregate_builds = 0;
  std::uint64_t attention_layout_evaluations = 0;
  std::uint64_t attention_unit_walks = 0;
};

// Enumerates every feasible mapping candidate of the workload on the
// hardware (design doc §5.1 candidate sets).  Throws DieModelError if the
// workload admits no mapping at all.
std::vector<Candidate> enumerate_candidates(const WorkloadQuantities& wl,
                                            const Hardware& hw);
// Diagnostic overload for tests that verify invariant attention work is
// hoisted across layout candidates.  Production callers use the two-argument
// overload above; with a null pointer no diagnostic counters are updated.
std::vector<Candidate> enumerate_candidates(const WorkloadQuantities& wl,
                                            const Hardware& hw,
                                            EnumerationDiagnostics* diagnostics);
// Index of the best candidate of a list under the SAME stage model and the
// SAME §5.1 tie-break as compile() (implemented in CostModel.cc; used by
// Mapping.cc's FFN fallback sub-mapping selection).
std::size_t pick_best_candidate(const std::vector<Candidate>& cands,
                                const Hardware& hw);

// Sound lower bound of one candidate's isolated makespan, used ONLY to
// prune the compile-time search inside pick_best_candidate (design doc
// §5.1 requires every feasible candidate to be compared by the full stage
// model; the bound guarantees the pruning never removes a winner).  Stage
// E3 replaced the stage A2 launch-serialized bound -- proven unsound by the
// stage E2 selection fuzz -- with per-slot busy sums, a lag-folded
// critical path and slot prefix+busy combinations; every component is
// provably <= evaluate_stages(stages, hw).total_seconds.  Declared at
// internal scope so the selection fuzz test can verify soundness and
// pruning equivalence directly.
double candidate_lower_bound(const Candidate& c, const Hardware& hw);

}  // namespace detail

}  // namespace DieModel
