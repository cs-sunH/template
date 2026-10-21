#pragma once

// Per-consumption-K OS GEMM access ledger. This is deliberately a local,
// single-tile counter: the caller supplies one actual (possibly tail) MxNxK
// tile and its K boundary flags. It does not choose mappings, model timing,
// own output slots, or infer when a final store can retire.

#include "EquivalentWork.hh"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace coarse_die::detail::gemm_os_access_counter {

using u64 = std::uint64_t;
using u128 = unsigned __int128;
using EqWork = equivalent_work::Work;

enum class StorageDType { Fp32, Bf16, Int4, Int32 };
enum class MatrixPath { Int4PE, StrictFp32Vec };
enum class Dataflow { OutputStationary, WeightResident };

struct Input {
  // Actual extents of one tile, not the nominal tm/tn/tk. In particular, odd
  // INT4 tails are packed independently for this tile.
  u64 m = 0;
  u64 n = 0;
  u64 k = 0;
  bool first_k = false;
  bool final_k = false;
  StorageDType a_storage = StorageDType::Fp32;
  StorageDType w_storage = StorageDType::Fp32;
  StorageDType output_storage = StorageDType::Fp32;
  MatrixPath matrix_path = MatrixPath::Int4PE;
  Dataflow dataflow = Dataflow::OutputStationary;
  double kappa_quant = 4.0;
  double kappa_acc = 2.0;
};

struct Result {
  // Physical HBM payload for this consumption. Input payload is read on every
  // OS K consumption; output payload is present only at final_k.
  u64 hbm_read_bytes = 0;
  u64 hbm_write_bytes = 0;

  // Action-role ledger. Load writes the raw storage inputs into SRAM, Compute
  // carries the actual conversion/consume/accumulate accesses, and Store reads
  // the final output payload. Aggregate SRAM R/W includes Load/Store; LOCAL is
  // only the Compute read+write traffic, matching Mapping.cc.
  u64 load_sram_write_bytes = 0;
  u64 compute_sram_read_bytes = 0;
  u64 compute_sram_write_bytes = 0;
  u64 store_sram_read_bytes = 0;
  u64 sram_read_bytes = 0;
  u64 sram_write_bytes = 0;
  u64 local_bytes = 0;

  u64 matrix_int4_ops = 0;
  EqWork vec_equivalent_ops;
  u64 conversion_elements = 0;
  u64 source_math_ops = 0;
};

namespace detail {

inline constexpr u128 max_u128() noexcept { return ~u128{0}; }

inline u128 checked_add(u128 a, u128 b, const char* field) {
  if (b > max_u128() - a)
    throw std::overflow_error(std::string("overflow: GEMM OS ") + field);
  return a + b;
}

inline u128 checked_mul(u128 a, u128 b, const char* field) {
  if (a != 0 && b > max_u128() / a)
    throw std::overflow_error(std::string("overflow: GEMM OS ") + field);
  return a * b;
}

inline u64 checked_u64(u128 value, const char* field) {
  if (value > static_cast<u128>(std::numeric_limits<u64>::max()))
    throw std::overflow_error(std::string("overflow: GEMM OS ") + field +
                              " exceeds uint64");
  return static_cast<u64>(value);
}

inline void check_dtype(StorageDType dtype, const char* role) {
  switch (dtype) {
    case StorageDType::Fp32:
    case StorageDType::Bf16:
    case StorageDType::Int4:
    case StorageDType::Int32:
      return;
  }
  throw std::invalid_argument(std::string("unsupported GEMM OS ") + role +
                              " storage dtype");
}

inline u128 bytes(u128 elements, StorageDType dtype) {
  switch (dtype) {
    case StorageDType::Fp32:
      return checked_mul(elements, 4, "FP32 payload");
    case StorageDType::Int32:
      return checked_mul(elements, 4, "INT32 payload");
    case StorageDType::Bf16:
      return checked_mul(elements, 2, "BF16 payload");
    case StorageDType::Int4:
      return elements / 2 + static_cast<u128>(elements % 2 != 0);
  }
  throw std::invalid_argument("unsupported GEMM OS storage dtype");
}

inline StorageDType execution_dtype(MatrixPath path) {
  switch (path) {
    case MatrixPath::Int4PE:
      return StorageDType::Int4;
    case MatrixPath::StrictFp32Vec:
      return StorageDType::Fp32;
  }
  throw std::invalid_argument("unsupported GEMM OS matrix path");
}

inline void require_kappas(const Input& in) {
  if (!std::isfinite(in.kappa_quant) || in.kappa_quant < 0.0)
    throw std::invalid_argument(
        "GEMM OS kappa_quant must be finite and nonnegative");
  if (!std::isfinite(in.kappa_acc) || in.kappa_acc < 0.0)
    throw std::invalid_argument(
        "GEMM OS kappa_acc must be finite and nonnegative");
}

inline u128 sum(u128 a, u128 b, u128 c, const char* field) {
  return checked_add(checked_add(a, b, field), c, field);
}

}  // namespace detail

// Count one output tile's one K consumption. For target INT4, each K emits a
// 4-byte-per-result INT32 partial and writes a 4-byte-per-result FP32
// accumulator; non-first K reads the old accumulator. Strict-FP32 uses the
// same byte ledger, but has no INT32 partial conversion or extra accumulator
// add Vec work, as in Mapping.cc's strict path.
inline Result count(const Input& in) {
  if (!in.m || !in.n || !in.k)
    throw std::invalid_argument("GEMM OS tile extents m,n,k must be positive");
  if (in.dataflow != Dataflow::OutputStationary)
    throw std::invalid_argument(
        "unsupported_dataflow: GemmOSAccessCounter accepts OS only");
  detail::check_dtype(in.a_storage, "A");
  detail::check_dtype(in.w_storage, "W");
  detail::check_dtype(in.output_storage, "output");
  detail::require_kappas(in);
  const StorageDType exec_dtype = detail::execution_dtype(in.matrix_path);
  const bool int4_path = in.matrix_path == MatrixPath::Int4PE;
  const bool convert_a = in.a_storage != exec_dtype;
  const bool convert_w = in.w_storage != exec_dtype;

  const u128 mn = detail::checked_mul(in.m, in.n, "tile M*N");
  const u128 a_elements = detail::checked_mul(in.m, in.k, "tile A elements");
  const u128 w_elements = detail::checked_mul(in.k, in.n, "tile W elements");
  const u128 math_ops = detail::checked_mul(
      2, detail::checked_mul(mn, in.k, "tile M*N*K"), "source math ops");
  const u128 a_storage_bytes = detail::bytes(a_elements, in.a_storage);
  const u128 w_storage_bytes = detail::bytes(w_elements, in.w_storage);
  const u128 a_exec_bytes = detail::bytes(a_elements, exec_dtype);
  const u128 w_exec_bytes = detail::bytes(w_elements, exec_dtype);
  const u128 output_bytes = detail::bytes(mn, in.output_storage);

  const u128 hbm_read = detail::checked_add(
      a_storage_bytes, w_storage_bytes, "HBM input payload");
  const u128 hbm_write = in.final_k ? output_bytes : 0;
  const u128 partial_bytes = int4_path
      ? detail::checked_mul(mn, 4, "INT32 partial bytes") : 0;
  const u128 accumulator_bytes = detail::checked_mul(
      mn, 4, "FP32 accumulator bytes");
  const u128 accumulator_read = in.first_k ? 0 : accumulator_bytes;

  // Conversion reads the just-loaded storage tile and writes one execution
  // tile. The target INT4 template also writes/reads one 4-byte scale per
  // converted input tile. Native inputs incur neither action.
  const u128 input_conversion_reads = detail::checked_add(
      convert_a ? a_storage_bytes : 0,
      convert_w ? w_storage_bytes : 0, "input conversion source reads");
  const u128 input_conversion_writes = detail::checked_add(
      convert_a ? a_exec_bytes : 0,
      convert_w ? w_exec_bytes : 0, "input conversion buffer writes");
  const u128 input_scale_bytes = int4_path
      ? detail::checked_mul(static_cast<u128>(convert_a) +
                                static_cast<u128>(convert_w),
                            4, "input conversion scales")
      : 0;
  const u128 input_exec_reads = detail::checked_add(
      a_exec_bytes, w_exec_bytes, "consumed execution input reads");

  // A cast is a final-K action only. INT4 output is packed per tile with ceil,
  // while its four-byte scale stays in the Compute SRAM write ledger; it is
  // not an extra HBM output payload in the current template.
  const bool cast_output = in.final_k &&
                           in.output_storage != StorageDType::Fp32;
  const u128 output_cast_read = cast_output ? accumulator_bytes : 0;
  const u128 output_scale_write =
      cast_output && in.output_storage == StorageDType::Int4 ? 4 : 0;

  const u128 compute_read = detail::sum(
      input_exec_reads, input_conversion_reads,
      detail::checked_add(
          detail::checked_add(partial_bytes, accumulator_read,
                              "Compute state reads"),
          detail::checked_add(input_scale_bytes, output_cast_read,
                              "Compute conversion reads"),
          "Compute reads"),
      "Compute reads");
  const u128 compute_write = detail::sum(
      detail::checked_add(input_conversion_writes, input_scale_bytes,
                          "Compute converted input writes"),
      detail::checked_add(partial_bytes, accumulator_bytes,
                          "Compute partial and accumulator writes"),
      cast_output ? detail::checked_add(output_bytes, output_scale_write,
                                        "Compute output cast writes") : 0,
      "Compute writes");

  const u128 load_sram_write = hbm_read;
  const u128 store_sram_read = hbm_write;
  const u128 sram_read = detail::checked_add(
      compute_read, store_sram_read, "total SRAM reads");
  const u128 sram_write = detail::checked_add(
      compute_write, load_sram_write, "total SRAM writes");
  const u128 local = detail::checked_add(
      compute_read, compute_write, "LOCAL traffic");

  u128 conversion_elements = 0;
  if (convert_a)
    conversion_elements = detail::checked_add(
        conversion_elements, a_elements, "conversion elements");
  if (convert_w)
    conversion_elements = detail::checked_add(
        conversion_elements, w_elements, "conversion elements");
  if (int4_path)
    conversion_elements = detail::checked_add(
        conversion_elements, mn, "accumulator conversion elements");
  if (cast_output)
    conversion_elements = detail::checked_add(
        conversion_elements, mn, "output conversion elements");

  EqWork vec_ops(int4_path ? 0 : math_ops);
  if (int4_path) {
    const u128 input_conversions = detail::checked_add(
        convert_a ? a_elements : 0, convert_w ? w_elements : 0,
        "input conversion elements");
    vec_ops += EqWork::weighted(input_conversions, in.kappa_quant);
    vec_ops += EqWork::weighted(mn, in.kappa_acc);
    if (!in.first_k) vec_ops += EqWork(mn);
  } else {
    // Mapping.cc charges BF16->FP32 at one eq-op/element and INT4 decoding at
    // kappa_quant; native FP32 needs no conversion.
    if (convert_a) {
      const double coeff = in.a_storage == StorageDType::Int4
                               ? in.kappa_quant : 1.0;
      vec_ops += EqWork::weighted(a_elements, coeff);
    }
    if (convert_w) {
      const double coeff = in.w_storage == StorageDType::Int4
                               ? in.kappa_quant : 1.0;
      vec_ops += EqWork::weighted(w_elements, coeff);
    }
  }
  if (cast_output) {
    const double coeff = in.output_storage == StorageDType::Int4
                             ? in.kappa_quant : 1.0;
    vec_ops += EqWork::weighted(mn, coeff);
  }

  Result out;
  out.hbm_read_bytes = detail::checked_u64(hbm_read, "HBM reads");
  out.hbm_write_bytes = detail::checked_u64(hbm_write, "HBM writes");
  out.load_sram_write_bytes =
      detail::checked_u64(load_sram_write, "Load SRAM writes");
  out.compute_sram_read_bytes =
      detail::checked_u64(compute_read, "Compute SRAM reads");
  out.compute_sram_write_bytes =
      detail::checked_u64(compute_write, "Compute SRAM writes");
  out.store_sram_read_bytes =
      detail::checked_u64(store_sram_read, "Store SRAM reads");
  out.sram_read_bytes = detail::checked_u64(sram_read, "SRAM reads");
  out.sram_write_bytes = detail::checked_u64(sram_write, "SRAM writes");
  out.local_bytes = detail::checked_u64(local, "LOCAL traffic");
  out.matrix_int4_ops = detail::checked_u64(
      int4_path ? math_ops : 0, "INT4 equivalent ops");
  out.vec_equivalent_ops = std::move(vec_ops);
  // The public counter boundary returns a uint64 whole part. Fractional work
  // remains the exact supported dyadic held by EquivalentWork::Work; never
  // combine it with the whole through a floating-point aggregate here.
  (void)out.vec_equivalent_ops.checked_u64_whole();
  out.conversion_elements =
      detail::checked_u64(conversion_elements, "conversion elements");
  out.source_math_ops =
      detail::checked_u64(math_ops, "source mathematical ops");
  return out;
}

}  // namespace coarse_die::detail::gemm_os_access_counter
