#include "test_support.hh"

#include "../src/GemmOSAccessCounter.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace coarse_die::detail {
std::vector<std::shared_ptr<Plan>> enumerate_mappings(
    const Hardware&, const nlohmann::json&);
}

using namespace coarse_die;
using namespace coarse_die_test;
namespace counter = coarse_die::detail::gemm_os_access_counter;
using u64 = std::uint64_t;
using u128 = unsigned __int128;

namespace {

template <class Exception, class Fn>
void check_throws(Fn&& fn, const std::string& message) {
  try {
    fn();
    throw std::runtime_error(message + " (did not throw)");
  } catch (const Exception&) {
  }
}

void check_work(const counter::EqWork& got, u64 whole,
                long double fraction, const std::string& what) {
  require(got.checked_u64_whole() == whole, what + " whole part mismatch");
  require(got.fraction() == fraction, what + " dyadic fraction mismatch");
}

counter::Input base_input() {
  counter::Input in;
  in.m = 2;
  in.n = 3;
  in.k = 5;
  in.first_k = true;
  in.final_k = true;
  in.a_storage = counter::StorageDType::Bf16;
  in.w_storage = counter::StorageDType::Int4;
  in.output_storage = counter::StorageDType::Int4;
  in.matrix_path = counter::MatrixPath::Int4PE;
  in.kappa_quant = 1.5;
  in.kappa_acc = 0.75;
  return in;
}

void hand_counted_fixtures() {
  // Odd packed W/C tiles, converted A tile and scale, K-first+K-final.
  const auto converted = counter::count(base_input());
  require(converted.hbm_read_bytes == 28, "converted tile HBM input payload");
  require(converted.hbm_write_bytes == 3, "odd INT4 C payload is ceil(MN/2)");
  require(converted.load_sram_write_bytes == 28,
          "Load writes exactly the raw input payload to SRAM");
  require(converted.compute_sram_read_bytes == 85,
          "conversion/scale/partial/final-cast Compute reads");
  require(converted.compute_sram_write_bytes == 64,
          "converted buffer/scale/partial/accumulator/output Compute writes");
  require(converted.store_sram_read_bytes == 3,
          "Store reads only the final packed output");
  require(converted.sram_read_bytes == 88 &&
              converted.sram_write_bytes == 92 && converted.local_bytes == 149,
          "aggregate SRAM and Compute-only LOCAL ledgers");
  require(converted.matrix_int4_ops == 60 && converted.source_math_ops == 60,
          "source mathematical work and INT4 PE work");
  require(converted.conversion_elements == 22,
          "input, partial-to-FP32 and output conversion elements");
  check_work(converted.vec_equivalent_ops, 28, 0.5L,
             "fractional target-INT4 Vec work");

  // Native target-INT4 inputs on a non-first, non-final K: partial read/write,
  // old FP32 accumulation read, no made-up C store, no input conversion.
  counter::Input middle;
  middle.m = 1;
  middle.n = 2;
  middle.k = 4;
  middle.first_k = false;
  middle.final_k = false;
  middle.a_storage = counter::StorageDType::Int4;
  middle.w_storage = counter::StorageDType::Int4;
  middle.output_storage = counter::StorageDType::Bf16;
  middle.matrix_path = counter::MatrixPath::Int4PE;
  middle.kappa_quant = 2.25;
  middle.kappa_acc = 0.5;
  const auto middle_count = counter::count(middle);
  require(middle_count.hbm_read_bytes == 6 &&
              middle_count.hbm_write_bytes == 0,
          "middle K reads its two packed inputs and writes no output");
  require(middle_count.compute_sram_read_bytes == 22 &&
              middle_count.compute_sram_write_bytes == 16 &&
              middle_count.local_bytes == 38,
          "non-first K reads old FP32 accumulation and partial");
  require(middle_count.sram_read_bytes == 22 &&
              middle_count.sram_write_bytes == 22 &&
              middle_count.store_sram_read_bytes == 0,
          "middle K has no Store action");
  require(middle_count.conversion_elements == 2,
          "native input middle K converts only its INT32 partial");
  check_work(middle_count.vec_equivalent_ops, 3, 0.0L,
             "accumulator conversion plus non-first add");

  // Strict FP32 path: matrix math is Vec work; BF16 conversion costs one,
  // INT4 decode uses kappa_quant, and there is no INT32 partial/add overhead.
  counter::Input strict;
  strict.m = 1;
  strict.n = 1;
  strict.k = 3;
  strict.first_k = true;
  strict.final_k = true;
  strict.a_storage = counter::StorageDType::Bf16;
  strict.w_storage = counter::StorageDType::Int4;
  strict.output_storage = counter::StorageDType::Bf16;
  strict.matrix_path = counter::MatrixPath::StrictFp32Vec;
  strict.kappa_quant = 1.25;
  strict.kappa_acc = 999.0;  // strict FP32 must not charge this coefficient.
  const auto strict_count = counter::count(strict);
  require(strict_count.hbm_read_bytes == 8 &&
              strict_count.hbm_write_bytes == 2,
          "strict FP32 input and final BF16 output payload");
  require(strict_count.compute_sram_read_bytes == 36 &&
              strict_count.compute_sram_write_bytes == 30 &&
              strict_count.local_bytes == 66,
          "strict FP32 conversion/accumulate/cast byte ledger");
  require(strict_count.matrix_int4_ops == 0 &&
              strict_count.source_math_ops == 6 &&
              strict_count.conversion_elements == 7,
          "strict FP32 source ops and conversion elements");
  check_work(strict_count.vec_equivalent_ops, 13, 0.75L,
             "strict FP32 matrix and input/output conversions");

  // INT32 is a supported 4-byte storage dtype. Strict FP32 converts it at one
  // eq-op/element; INT4PE conversion uses kappa_quant. Final INT32 cast remains
  // one eq-op/element on both paths.
  counter::Input int32_strict;
  int32_strict.m = 1;
  int32_strict.n = 1;
  int32_strict.k = 2;
  int32_strict.first_k = true;
  int32_strict.final_k = true;
  int32_strict.a_storage = counter::StorageDType::Int32;
  int32_strict.w_storage = counter::StorageDType::Int4;
  int32_strict.output_storage = counter::StorageDType::Int32;
  int32_strict.matrix_path = counter::MatrixPath::StrictFp32Vec;
  int32_strict.kappa_quant = 1.25;
  const auto int32_strict_count = counter::count(int32_strict);
  require(int32_strict_count.hbm_read_bytes == 9 &&
              int32_strict_count.hbm_write_bytes == 4,
          "strict INT32 A/output use four physical bytes per element");
  require(int32_strict_count.compute_sram_read_bytes == 29 &&
              int32_strict_count.compute_sram_write_bytes == 24 &&
              int32_strict_count.local_bytes == 53,
          "strict INT32 conversion and final-cast action bytes");
  require(int32_strict_count.conversion_elements == 5 &&
              int32_strict_count.source_math_ops == 4,
          "strict INT32 conversion and source-op elements");
  check_work(int32_strict_count.vec_equivalent_ops, 9, 0.5L,
             "strict INT32 and INT4 decode coefficients");

  counter::Input int32_int4 = int32_strict;
  int32_int4.matrix_path = counter::MatrixPath::Int4PE;
  int32_int4.kappa_acc = 0.75;
  const auto int32_int4_count = counter::count(int32_int4);
  require(int32_int4_count.hbm_read_bytes == 9 &&
              int32_int4_count.hbm_write_bytes == 4 &&
              int32_int4_count.compute_sram_read_bytes == 22 &&
              int32_int4_count.compute_sram_write_bytes == 17 &&
              int32_int4_count.local_bytes == 39,
          "target INT4 INT32 conversion and final-cast action bytes");
  require(int32_int4_count.matrix_int4_ops == 4 &&
              int32_int4_count.conversion_elements == 4,
          "target INT4 INT32 source/partial/output conversions");
  check_work(int32_int4_count.vec_equivalent_ops, 4, 0.25L,
             "target INT4 INT32 conversion uses kappa_quant");

  // A one-K native tile with FP32 output writes and stores the FP32 accumulator
  // directly. There is no final cast and no synthetic output-buffer write.
  counter::Input native_fp32;
  native_fp32.m = 1;
  native_fp32.n = 1;
  native_fp32.k = 1;
  native_fp32.first_k = true;
  native_fp32.final_k = true;
  native_fp32.a_storage = counter::StorageDType::Int4;
  native_fp32.w_storage = counter::StorageDType::Int4;
  native_fp32.output_storage = counter::StorageDType::Fp32;
  native_fp32.matrix_path = counter::MatrixPath::Int4PE;
  native_fp32.kappa_acc = 0.5;
  const auto native_count = counter::count(native_fp32);
  require(native_count.hbm_read_bytes == 2 &&
              native_count.hbm_write_bytes == 4 &&
              native_count.compute_sram_read_bytes == 6 &&
              native_count.compute_sram_write_bytes == 8 &&
              native_count.store_sram_read_bytes == 4,
          "native one-K FP32 output action ledger");
  require(native_count.conversion_elements == 1,
          "native one-K output only converts its INT32 partial");
  check_work(native_count.vec_equivalent_ops, 0, 0.5L,
             "native one-K accumulator conversion");
}

struct Totals {
  u128 hbm_read = 0;
  u128 hbm_write = 0;
  u128 load_sram_write = 0;
  u128 compute_sram_read = 0;
  u128 compute_sram_write = 0;
  u128 store_sram_read = 0;
  u128 sram_read = 0;
  u128 sram_write = 0;
  u128 local = 0;
  u128 int4_ops = 0;
  u128 conversion_elements = 0;
  u128 source_math_ops = 0;
  counter::EqWork vec_ops;

  void add(const counter::Result& row) {
    hbm_read += row.hbm_read_bytes;
    hbm_write += row.hbm_write_bytes;
    load_sram_write += row.load_sram_write_bytes;
    compute_sram_read += row.compute_sram_read_bytes;
    compute_sram_write += row.compute_sram_write_bytes;
    store_sram_read += row.store_sram_read_bytes;
    sram_read += row.sram_read_bytes;
    sram_write += row.sram_write_bytes;
    local += row.local_bytes;
    int4_ops += row.matrix_int4_ops;
    conversion_elements += row.conversion_elements;
    source_math_ops += row.source_math_ops;
    vec_ops += row.vec_equivalent_ops;
  }
};

std::vector<u64> extents(u64 length, u64 tile) {
  std::vector<u64> result;
  for (u64 offset = 0; offset < length; offset += tile)
    result.push_back(std::min(tile, length - offset));
  return result;
}

u64 summary_u64(const Plan& plan, const char* key) {
  if (!plan.summary.contains(key))
    throw std::runtime_error(std::string("candidate summary missing ") + key);
  return plan.summary.at(key).get<u64>();
}

void check_summary(const Plan& plan, const Totals& total) {
  require(summary_u64(plan, "hbm_read_bytes_total") ==
              static_cast<u64>(total.hbm_read), "compiled GEMM HBM read sum");
  require(summary_u64(plan, "hbm_write_bytes_total") ==
              static_cast<u64>(total.hbm_write), "compiled GEMM HBM write sum");
  require(summary_u64(plan, "sram_read_bytes") ==
              static_cast<u64>(total.sram_read), "compiled GEMM SRAM read sum");
  require(summary_u64(plan, "sram_write_bytes") ==
              static_cast<u64>(total.sram_write), "compiled GEMM SRAM write sum");
  require(summary_u64(plan, "local_bus_bytes") ==
              static_cast<u64>(total.local), "compiled GEMM LOCAL sum");
  require(summary_u64(plan, "executed_int4_ops") ==
              static_cast<u64>(total.int4_ops), "compiled GEMM INT4-op sum");
  require(summary_u64(plan, "conversion_elements") ==
              static_cast<u64>(total.conversion_elements),
          "compiled GEMM conversion-element sum");
  require(summary_u64(plan, "source_matrix_ops") ==
              static_cast<u64>(total.source_math_ops),
          "compiled GEMM source-math sum");

  const auto& summary = plan.summary;
  const u64 whole = summary.contains("vec_equivalent_ops_whole")
                        ? summary.at("vec_equivalent_ops_whole").get<u64>()
                        : summary.at("vec_equivalent_ops").get<u64>();
  const long double fraction = summary.contains("vec_equivalent_ops_fraction")
                                   ? equivalent_fraction_from_json(
                                         summary.at("vec_equivalent_ops_fraction"))
                                   : 0.0L;
  require(total.vec_ops.checked_u64_whole() == whole,
          "compiled GEMM Vec whole part");
  require(total.vec_ops.fraction() == fraction,
          "compiled GEMM exact dyadic Vec fraction");
}

void compare_with_compiled_fixed_tiles(const std::string& profile,
                                       counter::StorageDType a,
                                       counter::StorageDType w,
                                       counter::StorageDType output) {
  auto hardware_json = read_json(std::string(COARSE_DIE_SOURCE_DIR) +
                                 "/configs/example_hardware.json");
  hardware_json["performance_profile"]["kappa_quant_eqops"] = 1.25;
  hardware_json["performance_profile"]["kappa_acc_convert_eqops"] = 0.75;
  const Hardware hardware = resolve_hardware(hardware_json);
  const auto dtype_name = [](counter::StorageDType dtype) {
    switch (dtype) {
      case counter::StorageDType::Fp32: return "fp32";
      case counter::StorageDType::Bf16: return "bf16";
      case counter::StorageDType::Int4: return "int4";
      case counter::StorageDType::Int32: return "int32";
    }
    throw std::runtime_error("bad test dtype");
  };
  const json task{{"op_kind", "gemm"},
                  {"tensor_shapes", {{"m", 17}, {"n", 33}, {"k", 65}}},
                  {"storage_dtype", {{"activations", dtype_name(a)},
                                     {"weights", dtype_name(w)},
                                     {"output", dtype_name(output)}}},
                  {"execution_profile", profile},
                  {"layer_instances", 1},
                  {"weight_passes", 1}};
  const auto candidates = coarse_die::detail::enumerate_mappings(hardware, task);
  require(!candidates.empty(), "compiled GEMM generated candidates");

  bool saw_os = false;
  bool saw_tail = false;
  for (const auto& candidate : candidates) {
    if (candidate->summary.at("dataflow_id") != "output_stationary_v2")
      continue;
    saw_os = true;
    counter::Input prototype;
    prototype.a_storage = a;
    prototype.w_storage = w;
    prototype.output_storage = output;
    prototype.matrix_path = profile == "int4_matrix_vec_fp32_assumed_v1"
                                ? counter::MatrixPath::Int4PE
                                : counter::MatrixPath::StrictFp32Vec;
    prototype.kappa_quant = 1.25;
    prototype.kappa_acc = 0.75;
    const auto ms = extents(17, summary_u64(*candidate, "tm"));
    const auto ns = extents(33, summary_u64(*candidate, "tn"));
    const auto ks = extents(65, summary_u64(*candidate, "tk"));
    saw_tail = saw_tail || ms.size() > 1 || ns.size() > 1 || ks.size() > 1;
    Totals total;
    for (std::size_t mi = 0; mi < ms.size(); ++mi) {
      for (std::size_t ni = 0; ni < ns.size(); ++ni) {
        for (std::size_t ki = 0; ki < ks.size(); ++ki) {
          counter::Input tile = prototype;
          tile.m = ms[mi];
          tile.n = ns[ni];
          tile.k = ks[ki];
          tile.first_k = ki == 0;
          tile.final_k = ki + 1 == ks.size();
          total.add(counter::count(tile));
        }
      }
    }
    check_summary(*candidate, total);
  }
  require(saw_os, "compiled candidate enumerator produced an OS mapping");
  require(saw_tail, "compiled candidate comparison exercised M/N/K tails");
}

void validation_and_overflow() {
  auto invalid_dataflow = base_input();
  invalid_dataflow.dataflow = counter::Dataflow::WeightResident;
  check_throws<std::invalid_argument>(
      [&] { (void)counter::count(invalid_dataflow); },
      "WR is explicitly unsupported by the OS counter");

  auto invalid_dtype = base_input();
  invalid_dtype.a_storage = static_cast<counter::StorageDType>(99);
  check_throws<std::invalid_argument>(
      [&] { (void)counter::count(invalid_dtype); },
      "unknown storage dtype is rejected");

  auto overflow = base_input();
  overflow.m = 1;
  overflow.n = 1;
  overflow.k = u64{1} << 62;
  overflow.a_storage = counter::StorageDType::Fp32;
  overflow.w_storage = counter::StorageDType::Fp32;
  overflow.output_storage = counter::StorageDType::Fp32;
  overflow.matrix_path = counter::MatrixPath::StrictFp32Vec;
  check_throws<std::overflow_error>(
      [&] { (void)counter::count(overflow); },
      "byte payload beyond uint64 is rejected after checked intermediate math");

  auto huge_kappa = base_input();
  huge_kappa.m = huge_kappa.n = huge_kappa.k = 1;
  huge_kappa.a_storage = counter::StorageDType::Bf16;
  huge_kappa.w_storage = counter::StorageDType::Int4;
  huge_kappa.kappa_quant = std::numeric_limits<double>::max();
  check_throws<std::overflow_error>(
      [&] { (void)counter::count(huge_kappa); },
      "weighted equivalent work beyond uint64 output is rejected");

  // Keep an exactly representable dyadic fraction beside a whole part above
  // 2^53. The aggregate long-double conversion must reject the lost fraction;
  // callers can consume the exact whole/fraction fields separately.
  counter::Input large_whole;
  large_whole.m = 1;
  large_whole.n = 1;
  large_whole.k = u64{1} << 53;
  large_whole.first_k = true;
  large_whole.final_k = true;
  large_whole.a_storage = counter::StorageDType::Fp32;
  large_whole.w_storage = counter::StorageDType::Fp32;
  large_whole.output_storage = counter::StorageDType::Int4;
  large_whole.matrix_path = counter::MatrixPath::StrictFp32Vec;
  large_whole.kappa_quant = std::ldexp(1.0, -20);
  const auto large_count = counter::count(large_whole);
  check_work(large_count.vec_equivalent_ops, u64{1} << 54,
             std::ldexp(1.0L, -20), "large whole plus tiny exact fraction");
  check_throws<std::overflow_error>(
      [&] { (void)large_count.vec_equivalent_ops.as_long_double(); },
      "large whole plus fraction cannot silently merge through long double");
}

void gemm_os_access_counter() {
  hand_counted_fixtures();
  compare_with_compiled_fixed_tiles(
      "int4_matrix_vec_fp32_assumed_v1", counter::StorageDType::Bf16,
      counter::StorageDType::Fp32, counter::StorageDType::Int4);
  compare_with_compiled_fixed_tiles(
      "fp32_vec_matrix_assumed_v1", counter::StorageDType::Int4,
      counter::StorageDType::Bf16, counter::StorageDType::Bf16);
  compare_with_compiled_fixed_tiles(
      "int4_matrix_vec_fp32_assumed_v1", counter::StorageDType::Int32,
      counter::StorageDType::Bf16, counter::StorageDType::Int32);
  compare_with_compiled_fixed_tiles(
      "fp32_vec_matrix_assumed_v1", counter::StorageDType::Int32,
      counter::StorageDType::Int4, counter::StorageDType::Int32);
  validation_and_overflow();
}

}  // namespace

int main() {
  return run_test("gemm_os_access_counter_test", gemm_os_access_counter);
}
