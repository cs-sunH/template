#pragma once

// Scratch-only bounded exact single-owner lifecycle kernel.  Facts are read
// from the existing public Runtime driver's JSON report; this file does not
// synthesize resource work, bytes, operations, or private-time formulas.

#include "../../die/src/GemmOSGeometry.hh"
#include "../os-fixedwidth-13-20261014/checked_rational256.hh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include <json/json.hpp>

namespace solo13 {

using Rational = os_fixedwidth::Rational256;
using u64 = std::uint64_t;
using u128 = unsigned __int128;
namespace geom = coarse_die::detail::gemm_os_geometry;
using json = nlohmann::json;

constexpr std::size_t kStages = 3;
constexpr std::size_t kRows = 13;
constexpr std::size_t kMaxClasses = 32;
constexpr std::size_t kQueueSlots = 2;
constexpr std::size_t kHistory = 256;
constexpr u64 kGlobalEventCap = 256;

inline std::string u128_string(u128 value) {
  if (value == 0) return "0";
  std::string result;
  while (value != 0) {
    result.push_back(static_cast<char>('0' + value % 10));
    value /= 10;
  }
  std::reverse(result.begin(), result.end());
  return result;
}

inline u128 parse_u128(std::string_view text) {
  if (text.empty()) throw std::invalid_argument("empty u128");
  constexpr u128 max = ~u128{0};
  u128 result = 0;
  for (char c : text) {
    if (c < '0' || c > '9') throw std::invalid_argument("invalid u128 decimal");
    const u128 digit = static_cast<unsigned>(c - '0');
    if (result > (max - digit) / 10)
      throw std::overflow_error("u128 decimal overflow");
    result = result * 10 + digit;
  }
  return result;
}

inline u128 checked_add(u128 a, u128 b, const char* what) {
  if (b > ~u128{0} - a) throw std::overflow_error(what);
  return a + b;
}

inline u128 checked_mul(u128 a, u128 b, const char* what) {
  if (a != 0 && b > ~u128{0} / a) throw std::overflow_error(what);
  return a * b;
}

inline Rational r_integer(std::string_view value) {
  Rational result;
  std::string error;
  if (!Rational::from_integer(value, &result, &error))
    throw std::overflow_error(error);
  return result;
}

inline Rational r_fraction(std::string_view value) {
  Rational result;
  std::string error;
  if (!Rational::from_fraction(value, &result, &error))
    throw std::overflow_error(error);
  return result;
}

inline Rational r_double(double value) {
  Rational result;
  std::string error;
  if (!Rational::from_double(value, &result, &error))
    throw std::overflow_error(error);
  return result;
}

inline Rational r_add(const Rational& a, const Rational& b) {
  Rational result;
  std::string error;
  if (!Rational::add(a, b, &result, &error)) throw std::overflow_error(error);
  return result;
}

inline Rational r_sub(const Rational& a, const Rational& b) {
  Rational result;
  std::string error;
  if (!Rational::subtract(a, b, &result, &error)) throw std::overflow_error(error);
  return result;
}

inline Rational r_mul(const Rational& a, const Rational& b) {
  Rational result;
  std::string error;
  if (!Rational::multiply(a, b, &result, &error)) throw std::overflow_error(error);
  return result;
}

inline Rational r_div(const Rational& a, const Rational& b) {
  Rational result;
  std::string error;
  if (!Rational::divide(a, b, &result, &error)) throw std::overflow_error(error);
  return result;
}

inline int r_compare(const Rational& a, const Rational& b) {
  int result = 0;
  std::string error;
  if (!Rational::compare(a, b, &result, &error)) throw std::overflow_error(error);
  return result;
}

inline Rational r_u128(u128 value) { return r_integer(u128_string(value)); }

inline std::string json_number_text(const json& value) {
  if (value.is_string()) return value.get<std::string>();
  return value.dump();
}

inline Rational rational_json_number(const json& value) {
  if (value.is_number_float()) return r_double(value.get<double>());
  if (value.is_number_integer() || value.is_number_unsigned() || value.is_string())
    return r_integer(json_number_text(value));
  throw std::invalid_argument("expected numeric rational fact");
}

inline std::string pow2_decimal(unsigned exponent) {
  os_fixedwidth::UInt768 value;
  value.limb[0] = 1;
  if (!value.shift_left(exponent)) throw std::overflow_error("dyadic exponent overflow");
  return value.decimal();
}

inline Rational dyadic_json_exact(const json& value) {
  if (value.is_null()) return Rational{};
  const std::string sig = json_number_text(value.at("significand"));
  const int exponent = value.at("exponent").get<int>();
  if (exponent < 0) {
    const unsigned shift = static_cast<unsigned>(-static_cast<long long>(exponent));
    return r_fraction(sig + "/" + pow2_decimal(shift));
  }
  const Rational base = r_integer(sig);
  return r_mul(base, r_integer(pow2_decimal(static_cast<unsigned>(exponent))));
}

inline std::string stage_name(std::size_t stage) {
  static constexpr const char* names[] = {"load", "compute", "store"};
  return names[stage];
}

struct ActivityFact {
  std::array<Rational, kRows> work{};
  Rational canonical_q{};
  Rational private_time{};
  std::array<std::array<Rational, 2>, 2> hbm{};
  Rational int4_ops{};
  Rational vec_eqops{};
};

struct ClassFact {
  std::array<ActivityFact, kStages> stage{};
};

struct Facts {
  geom::Input input;
  geom::GemmOSGeometry geometry;
  std::array<ClassFact, kMaxClasses> classes{};
  std::size_t class_count = 0;
  std::array<Rational, kRows> capacities{};

  explicit Facts(const json& report)
      : input(parse_geometry_input(report)), geometry(input) {
    const json& g = report.at("geometry");
    if (input.core_count != 1)
      throw std::invalid_argument("scratch kernel supports core_count=1 only");
    if (geometry.class_count() == 0 || geometry.class_count() > kMaxClasses)
      throw std::invalid_argument("geometry access classes outside 1..32");
    const json& class_json = g.at("classes");
    if (!class_json.is_array() || class_json.size() != geometry.class_count())
      throw std::invalid_argument("facts class count differs from GemmOSGeometry");
    class_count = geometry.class_count();
    for (std::size_t i = 0; i < class_count; ++i) {
      validate_geometry_class(geometry.access_class(i), class_json.at(i));
      for (std::size_t s = 0; s < kStages; ++s)
        classes[i].stage[s] = parse_activity(
            class_json.at(i).at(stage_name(s) + "_fact"));
    }
    const json& synthetic = report.at("synthetic_hardware");
    const json& cap = synthetic.at("hardware_capacities_13");
    if (!cap.is_array() || cap.size() != kRows)
      throw std::invalid_argument("facts capacity vector is not 13 domains");
    for (std::size_t r = 0; r < kRows; ++r) capacities[r] = rational_json_number(cap.at(r));
    for (std::size_t i = 0; i < class_count; ++i) {
      const auto& facts_cap = class_json.at(i).at("hardware_capacities_13");
      if (!facts_cap.is_array() || facts_cap.size() != kRows)
        throw std::invalid_argument("class facts capacity vector malformed");
      for (std::size_t r = 0; r < kRows; ++r) {
        if (!rational_json_number(facts_cap.at(r)).equal(capacities[r]))
          throw std::invalid_argument("class and report capacities differ");
      }
    }
  }

  std::size_t class_for_wave(u128 wave) const {
    if (wave >= geometry.consumption_wave_count())
      throw std::out_of_range("consumption wave outside geometry");
    const u64 k_tiles = geometry.k_tiles();
    const u128 q = wave / k_tiles;
    const u64 k = static_cast<u64>(wave % k_tiles);
    return geometry.wave_fact(q, k).class_index;
  }

  const ActivityFact& activity(std::size_t stage, u128 wave) const {
    return classes.at(class_for_wave(wave)).stage.at(stage);
  }

  bool is_final(u128 wave) const {
    return geometry.access_class(class_for_wave(wave)).final_k;
  }

 private:
  static geom::Input parse_geometry_input(const json& report) {
    const json& g = report.at("geometry");
    geom::Input in;
    in.M = g.at("M").get<u64>(); in.N = g.at("N").get<u64>();
    in.K = g.at("K").get<u64>(); in.tm = g.at("tm").get<u64>();
    in.tn = g.at("tn").get<u64>(); in.tk = g.at("tk").get<u64>();
    in.core_count = g.at("core_count").get<u64>();
    const auto dtype = [](const std::string& name) {
      if (name == "fp32") return coarse_die::detail::gemm_os_access_counter::StorageDType::Fp32;
      if (name == "bf16") return coarse_die::detail::gemm_os_access_counter::StorageDType::Bf16;
      if (name == "int4") return coarse_die::detail::gemm_os_access_counter::StorageDType::Int4;
      if (name == "int32") return coarse_die::detail::gemm_os_access_counter::StorageDType::Int32;
      throw std::invalid_argument("unsupported storage dtype in facts report");
    };
    in.a_storage = dtype(g.at("a_storage").get<std::string>());
    in.w_storage = dtype(g.at("w_storage").get<std::string>());
    in.output_storage = dtype(g.at("output_storage").get<std::string>());
    const std::string path = g.at("path").get<std::string>();
    if (path == "strict_fp32_vec")
      in.matrix_path = coarse_die::detail::gemm_os_access_counter::MatrixPath::StrictFp32Vec;
    else if (path == "int4_pe")
      in.matrix_path = coarse_die::detail::gemm_os_access_counter::MatrixPath::Int4PE;
    else throw std::invalid_argument("unsupported matrix path in facts report");
    in.kappa_quant = g.at("kappa_quant").get<double>();
    in.kappa_acc = g.at("kappa_acc").get<double>();
    return in;
  }

  static void validate_geometry_class(const geom::AccessClass& expected,
                                      const json& found) {
    if (found.at("index").get<std::size_t>() != expected.index ||
        found.at("m_tile_begin").get<u64>() != expected.m_tile_begin ||
        found.at("m_tile_count").get<u64>() != expected.m_tile_count ||
        found.at("n_tile_begin").get<u64>() != expected.n_tile_begin ||
        found.at("n_tile_count").get<u64>() != expected.n_tile_count ||
        found.at("k_tile_begin").get<u64>() != expected.k_tile_begin ||
        found.at("k_tile_count").get<u64>() != expected.k_tile_count ||
        found.at("actual_m").get<u64>() != expected.m ||
        found.at("actual_n").get<u64>() != expected.n ||
        found.at("actual_k").get<u64>() != expected.k ||
        found.at("m_tail").get<bool>() != expected.m_tail ||
        found.at("n_tail").get<bool>() != expected.n_tail ||
        found.at("k_tail").get<bool>() != expected.k_tail ||
        found.at("first_k").get<bool>() != expected.first_k ||
        found.at("final_k").get<bool>() != expected.final_k ||
        parse_u128(json_number_text(found.at("repeat_count"))) != expected.repeat_count)
      throw std::invalid_argument("class facts do not match GemmOSGeometry");
  }

  static ActivityFact parse_activity(const json& activity) {
    ActivityFact result;
    const json& work = activity.at("resource_work_13");
    if (!work.is_array() || work.size() != kRows)
      throw std::invalid_argument("activity facts missing 13 resource-work rows");
    for (std::size_t r = 0; r < kRows; ++r)
      result.work[r] = rational_json_number(work.at(r));
    result.canonical_q = rational_json_number(activity.at("canonical_q"));
    result.private_time = rational_json_number(activity.at("private_time_s"));
    const json& payload = activity.at("hbm_row_payload_bytes");
    if (!payload.is_array() || payload.size() != 2)
      throw std::invalid_argument("activity facts missing two HBM rows");
    for (std::size_t row = 0; row < 2; ++row) {
      if (!payload.at(row).is_array() || payload.at(row).size() != 2)
        throw std::invalid_argument("activity HBM payload missing directions");
      for (std::size_t dir = 0; dir < 2; ++dir)
        result.hbm[row][dir] = rational_json_number(payload.at(row).at(dir));
    }
    const json& typed = activity.at("typed_ops");
    result.int4_ops = rational_json_number(typed.at("int4_ops"));
    result.vec_eqops = rational_json_number(typed.at("vec_eqops_whole"));
    if (typed.contains("vec_eqops_fraction"))
      result.vec_eqops = r_add(result.vec_eqops,
                               dyadic_json_exact(typed.at("vec_eqops_fraction")));
    if (r_compare(result.canonical_q, Rational{}) <= 0)
      throw std::invalid_argument("activity canonical Q must be positive");
    if (r_compare(result.private_time, Rational{}) < 0)
      throw std::invalid_argument("activity private time must be nonnegative");
    return result;
  }
};

struct Ledger {
  Rational elapsed{};
  std::array<Rational, kRows> service{};
  std::array<Rational, kRows> busy{};
  std::array<std::array<Rational, 2>, 2> hbm_payload{};
  Rational int4_ops{};
  Rational vec_eqops{};
  Rational compute_busy{};
  std::array<Rational, 2> hbm_row_busy{};
  Rational hbm_busy{};
};

inline Ledger ledger_delta(const Ledger& newer, const Ledger& older) {
  Ledger out;
  out.elapsed = r_sub(newer.elapsed, older.elapsed);
  for (std::size_t r = 0; r < kRows; ++r) {
    out.service[r] = r_sub(newer.service[r], older.service[r]);
    out.busy[r] = r_sub(newer.busy[r], older.busy[r]);
  }
  for (std::size_t row = 0; row < 2; ++row)
    for (std::size_t dir = 0; dir < 2; ++dir)
      out.hbm_payload[row][dir] = r_sub(newer.hbm_payload[row][dir],
                                        older.hbm_payload[row][dir]);
  out.int4_ops = r_sub(newer.int4_ops, older.int4_ops);
  out.vec_eqops = r_sub(newer.vec_eqops, older.vec_eqops);
  out.compute_busy = r_sub(newer.compute_busy, older.compute_busy);
  for (std::size_t r = 0; r < 2; ++r)
    out.hbm_row_busy[r] = r_sub(newer.hbm_row_busy[r], older.hbm_row_busy[r]);
  out.hbm_busy = r_sub(newer.hbm_busy, older.hbm_busy);
  return out;
}

inline void ledger_add_repeat(Ledger* to, const Ledger& unit, u128 repeat) {
  const std::string repeat_text = u128_string(repeat);
  const auto scale = [&repeat_text](const Rational& value) {
    Rational result;
    std::string error;
    if (!Rational::multiply_repeat(value, repeat_text, &result, &error))
      throw std::overflow_error(error);
    return result;
  };
  to->elapsed = r_add(to->elapsed, scale(unit.elapsed));
  for (std::size_t r = 0; r < kRows; ++r) {
    to->service[r] = r_add(to->service[r], scale(unit.service[r]));
    to->busy[r] = r_add(to->busy[r], scale(unit.busy[r]));
  }
  for (std::size_t row = 0; row < 2; ++row)
    for (std::size_t dir = 0; dir < 2; ++dir)
      to->hbm_payload[row][dir] = r_add(to->hbm_payload[row][dir],
                                        scale(unit.hbm_payload[row][dir]));
  to->int4_ops = r_add(to->int4_ops, scale(unit.int4_ops));
  to->vec_eqops = r_add(to->vec_eqops, scale(unit.vec_eqops));
  to->compute_busy = r_add(to->compute_busy, scale(unit.compute_busy));
  for (std::size_t r = 0; r < 2; ++r)
    to->hbm_row_busy[r] = r_add(to->hbm_row_busy[r], scale(unit.hbm_row_busy[r]));
  to->hbm_busy = r_add(to->hbm_busy, scale(unit.hbm_busy));
}

struct Active {
  bool present = false;
  u128 wave = 0;
  Rational progress{};
};

struct OwnerState {
  std::array<Active, kStages> active{};
  std::array<u128, kQueueSlots> loaded{};
  std::size_t loaded_count = 0;
  std::array<u128, kQueueSlots> computed{};
  std::size_t computed_count = 0;
  u128 next_load = 0;
  u128 retired = 0;
  bool finished = false;
};

struct Snapshot {
  OwnerState owner;
  Ledger ledger;
};

class Kernel {
 public:
  explicit Kernel(const json& facts, bool enable_identity_skips = true)
      : facts_(facts), enable_identity_skips_(enable_identity_skips) {
    for (std::size_t r = 0; r < kRows; ++r) capacity_[r] = facts_.capacities[r];
    total_waves_ = facts_.geometry.consumption_wave_count();
    total_outputs_ = facts_.geometry.output_tile_count();
    k_tiles_ = facts_.geometry.k_tiles();
    n_tiles_ = facts_.geometry.n_tiles();
    m_tiles_ = facts_.geometry.m_tiles();
    if (facts_.input.core_count != 1)
      throw std::invalid_argument("single-core geometry required");
  }

  enum class Status { Passed, Paused, FailedClosed };

  Status run_until(const Rational* stop = nullptr) {
    if (stop != nullptr && r_compare(*stop, ledger_.elapsed) < 0) {
      error_ = "external stop reverses lifecycle clock";
      return Status::FailedClosed;
    }
    try {
      stop_ = stop == nullptr ? std::optional<Rational>{}
                              : std::optional<Rational>{*stop};
      settle();
      while (!owner_.finished) {
        if (stop_ && r_compare(ledger_.elapsed, *stop_) == 0)
          return Status::Paused;
        if (enable_identity_skips_) try_identity_skip();
        if (owner_.finished) break;
        if (stop_ && r_compare(ledger_.elapsed, *stop_) == 0)
          return Status::Paused;
        event();
      }
      validate_complete();
      return Status::Passed;
    } catch (const std::exception& error) {
      error_ = error.what();
      return Status::FailedClosed;
    }
  }

  Status resume_until(const Rational* stop = nullptr,
                     const std::array<Rational, kRows>* capacities = nullptr) {
    if (stop != nullptr && r_compare(*stop, ledger_.elapsed) < 0) {
      error_ = "external stop reverses lifecycle clock";
      return Status::FailedClosed;
    }
    if (capacities != nullptr) {
      for (std::size_t r = 0; r < kRows; ++r) {
        if (r_compare((*capacities)[r], Rational{}) < 0) {
          error_ = "capacity vector contains a negative rate";
          return Status::FailedClosed;
        }
      }
      capacity_ = *capacities;
      history_count_ = 0;  // capacities guard the old exact identity return
      ++capacity_invalidations_;
    }
    return run_until(stop);
  }

  u64 events() const noexcept { return events_; }
  u128 skipped_waves() const noexcept { return skipped_waves_; }
  u128 skipped_outputs() const noexcept { return skipped_outputs_; }
  const Ledger& ledger() const noexcept { return ledger_; }
  const OwnerState& owner() const noexcept { return owner_; }
  const std::string& error() const noexcept { return error_; }
  std::size_t history_count() const noexcept { return history_count_; }
  u64 peak_active() const noexcept { return peak_active_; }
  u64 peak_input() const noexcept { return peak_input_; }
  u64 peak_output() const noexcept { return peak_output_; }
  u64 nonfinal_compute_at_full_output() const noexcept { return nonfinal_full_output_; }
  u64 final_compute_blocked_on_full_output() const noexcept { return final_full_output_waits_; }
  bool identity_skips_enabled() const noexcept { return enable_identity_skips_; }
  u64 capacity_invalidations() const noexcept { return capacity_invalidations_; }
  const std::array<Rational, kRows>& capacities() const noexcept { return capacity_; }
  const Facts& facts() const noexcept { return facts_; }

 private:
  struct KeyRef { u128 wave = 0; };

  Facts facts_;
  OwnerState owner_{};
  Ledger ledger_{};
  std::array<Rational, kRows> capacity_{};
  u128 total_waves_ = 0, total_outputs_ = 0;
  u64 k_tiles_ = 0, n_tiles_ = 0, m_tiles_ = 0;
  bool enable_identity_skips_ = true;
  u64 events_ = 0;
  u128 skipped_waves_ = 0, skipped_outputs_ = 0;
  std::array<Snapshot, kHistory> history_{};
  std::size_t history_count_ = 0;
  std::optional<Rational> stop_;
  std::string error_;
  u64 peak_active_ = 0, peak_input_ = 0, peak_output_ = 0;
  u64 nonfinal_full_output_ = 0;
  u128 last_blocked_final_wave_ = 0;
  bool has_blocked_final_wave_ = false;
  u64 final_full_output_waits_ = 0;
  u64 capacity_invalidations_ = 0;

  std::pair<u64, u64> occupancy() const {
    u64 input = static_cast<u64>(owner_.loaded_count);
    if (owner_.active[0].present) ++input;
    if (owner_.active[1].present) ++input;
    u64 output = static_cast<u64>(owner_.computed_count);
    if (owner_.active[2].present) ++output;
    if (owner_.active[1].present && facts_.is_final(owner_.active[1].wave)) ++output;
    return {input, output};
  }

  void settle() {
    for (;;) {
      bool changed = false;
      if (owner_.computed_count > 0 && !owner_.active[2].present) {
        owner_.active[2] = {true, owner_.computed[0], Rational{}};
        for (std::size_t i = 1; i < owner_.computed_count; ++i)
          owner_.computed[i - 1] = owner_.computed[i];
        --owner_.computed_count;
        changed = true;
      }
      if (owner_.loaded_count > 0 && !owner_.active[1].present) {
        const u128 wave = owner_.loaded[0];
        const auto [input, output] = occupancy();
        (void)input;
        if (facts_.is_final(wave) && output >= kQueueSlots) {
          if (!has_blocked_final_wave_ || last_blocked_final_wave_ != wave) {
            ++final_full_output_waits_;
            last_blocked_final_wave_ = wave;
            has_blocked_final_wave_ = true;
          }
        }
        if (!facts_.is_final(wave) || output < kQueueSlots) {
          for (std::size_t i = 1; i < owner_.loaded_count; ++i)
            owner_.loaded[i - 1] = owner_.loaded[i];
          --owner_.loaded_count;
          owner_.active[1] = {true, wave, Rational{}};
          if (!facts_.is_final(wave) && output == kQueueSlots)
            ++nonfinal_full_output_;
          changed = true;
        }
      }
      const auto [input, output] = occupancy();
      peak_input_ = std::max(peak_input_, input);
      peak_output_ = std::max(peak_output_, output);
      if (input > kQueueSlots || output > kQueueSlots)
        throw std::logic_error("two-slot readiness queue capacity exceeded");
      if (!owner_.active[0].present && owner_.next_load < total_waves_ &&
          input < kQueueSlots) {
        owner_.active[0] = {true, owner_.next_load, Rational{}};
        owner_.next_load = checked_add(owner_.next_load, 1, "next load cursor overflow");
        changed = true;
      }
      if (owner_.retired == total_outputs_ && no_live_work()) {
        owner_.finished = true;
        return;
      }
      if (!changed) return;
    }
  }

  bool no_live_work() const {
    return !owner_.active[0].present && !owner_.active[1].present &&
        !owner_.active[2].present && owner_.loaded_count == 0 &&
        owner_.computed_count == 0;
  }

  std::size_t refs(std::array<u128, 8>* output) const {
    std::size_t count = 0;
    for (const auto& active : owner_.active)
      if (active.present) (*output)[count++] = active.wave;
    for (std::size_t i = 0; i < owner_.loaded_count; ++i)
      (*output)[count++] = owner_.loaded[i];
    for (std::size_t i = 0; i < owner_.computed_count; ++i)
      (*output)[count++] = owner_.computed[i];
    if (owner_.next_load < total_waves_)
      (*output)[count++] = owner_.next_load;
    return count;
  }

  u128 anchor() const {
    std::array<u128, 8> waves{};
    const std::size_t count = refs(&waves);
    if (count == 0) return owner_.next_load;
    return *std::min_element(waves.begin(), waves.begin() + count);
  }

  bool class_signature_equal(u128 a, u128 b) const {
    const auto& x = facts_.geometry.access_class(facts_.class_for_wave(a));
    const auto& y = facts_.geometry.access_class(facts_.class_for_wave(b));
    if (x.m != y.m || x.n != y.n || x.k != y.k ||
        x.m_tail != y.m_tail || x.n_tail != y.n_tail || x.k_tail != y.k_tail ||
        x.first_k != y.first_k || x.final_k != y.final_k) return false;
    const std::size_t xi = x.index, yi = y.index;
    for (std::size_t s = 0; s < kStages; ++s) {
      const auto& xa = facts_.classes[xi].stage[s];
      const auto& ya = facts_.classes[yi].stage[s];
      if (!xa.canonical_q.equal(ya.canonical_q) ||
          !xa.private_time.equal(ya.private_time) ||
          !xa.int4_ops.equal(ya.int4_ops) || !xa.vec_eqops.equal(ya.vec_eqops))
        return false;
      for (std::size_t r = 0; r < kRows; ++r)
        if (!xa.work[r].equal(ya.work[r])) return false;
      for (std::size_t row = 0; row < 2; ++row)
        for (std::size_t dir = 0; dir < 2; ++dir)
          if (!xa.hbm[row][dir].equal(ya.hbm[row][dir])) return false;
    }
    return true;
  }

  u128 snapshot_anchor(const Snapshot& old) const {
    std::array<u128, 8> waves{};
    std::size_t count = 0;
    for (const auto& active : old.owner.active)
      if (active.present) waves[count++] = active.wave;
    for (std::size_t i = 0; i < old.owner.loaded_count; ++i)
      waves[count++] = old.owner.loaded[i];
    for (std::size_t i = 0; i < old.owner.computed_count; ++i)
      waves[count++] = old.owner.computed[i];
    if (old.owner.next_load < total_waves_) waves[count++] = old.owner.next_load;
    if (count == 0) return old.owner.next_load;
    return *std::min_element(waves.begin(), waves.begin() + count);
  }

  bool topology_return(const Snapshot& old, u128* shift) const {
    const u128 old_anchor = snapshot_anchor(old);
    const u128 new_anchor = anchor();
    if (new_anchor <= old_anchor) return false;
    const u128 d = new_anchor - old_anchor;
    const auto same_signed_difference = [](u128 a, u128 b, u128 c, u128 e) {
      const bool first_negative = a < b, second_negative = c < e;
      const u128 first_magnitude = first_negative ? b - a : a - b;
      const u128 second_magnitude = second_negative ? e - c : c - e;
      return first_negative == second_negative && first_magnitude == second_magnitude;
    };
    if (!same_signed_difference(old.owner.retired, old_anchor / k_tiles_,
                                owner_.retired, new_anchor / k_tiles_))
      return false;
    const bool old_has_next = old.owner.next_load < total_waves_;
    const bool new_has_next = owner_.next_load < total_waves_;
    if (old_has_next != new_has_next) return false;
    if (old_has_next && !class_signature_equal(old.owner.next_load, owner_.next_load))
      return false;
    std::array<u128, 8> old_refs{}, new_refs{};
    const std::size_t old_count = snapshot_refs(old, &old_refs);
    const std::size_t new_count = refs(&new_refs);
    if (old_count == 0 || new_count == 0) return false;
    for (std::size_t s = 0; s < kStages; ++s) {
      const auto& a = old.owner.active[s];
      const auto& b = owner_.active[s];
      if (a.present != b.present) return false;
      if (a.present && (b.wave < a.wave || b.wave - a.wave != d ||
                        !a.progress.equal(b.progress) ||
                        !class_signature_equal(a.wave, b.wave))) return false;
    }
    if (old.owner.loaded_count != owner_.loaded_count ||
        old.owner.computed_count != owner_.computed_count) return false;
    for (std::size_t i = 0; i < owner_.loaded_count; ++i) {
      if (owner_.loaded[i] < old.owner.loaded[i] ||
          owner_.loaded[i] - old.owner.loaded[i] != d ||
          !class_signature_equal(old.owner.loaded[i], owner_.loaded[i])) return false;
    }
    for (std::size_t i = 0; i < owner_.computed_count; ++i) {
      if (owner_.computed[i] < old.owner.computed[i] ||
          owner_.computed[i] - old.owner.computed[i] != d ||
          !class_signature_equal(old.owner.computed[i], owner_.computed[i])) return false;
    }
    if (owner_.next_load < old.owner.next_load ||
        owner_.next_load - old.owner.next_load != d) return false;
    *shift = d;
    return true;
  }

  std::size_t snapshot_refs(const Snapshot& old,
                            std::array<u128, 8>* output) const {
    std::size_t count = 0;
    for (const auto& active : old.owner.active)
      if (active.present) (*output)[count++] = active.wave;
    for (std::size_t i = 0; i < old.owner.loaded_count; ++i)
      (*output)[count++] = old.owner.loaded[i];
    for (std::size_t i = 0; i < old.owner.computed_count; ++i)
      (*output)[count++] = old.owner.computed[i];
    if (old.owner.next_load < total_waves_)
      (*output)[count++] = old.owner.next_load;
    return count;
  }

  struct RowWordEntry { u64 n_begin, n_count, k_begin, k_count; std::size_t cid; };

  std::size_t whole_row_word(u64 m_tile,
                             std::array<RowWordEntry, kMaxClasses>* word) const {
    if (m_tile >= m_tiles_) return 0;
    std::size_t count = 0;
    for (std::size_t i = 0; i < facts_.geometry.class_count(); ++i) {
      const auto& cls = facts_.geometry.access_class(i);
      if (m_tile >= cls.m_tile_begin &&
          m_tile - cls.m_tile_begin < cls.m_tile_count) {
        (*word)[count++] = {cls.n_tile_begin, cls.n_tile_count,
                            cls.k_tile_begin, cls.k_tile_count, i};
      }
    }
    std::sort(word->begin(), word->begin() + count,
              [](const RowWordEntry& a, const RowWordEntry& b) {
                if (a.n_begin != b.n_begin) return a.n_begin < b.n_begin;
                return a.k_begin < b.k_begin;
              });
    u64 n_cursor = 0, k_cursor = 0;
    u64 active_n_begin = 0, active_n_count = 0;
    for (std::size_t i = 0; i < count; ++i) {
      const auto& item = (*word)[i];
      if (i == 0 || item.n_begin != active_n_begin) {
        if (i != 0 && k_cursor != k_tiles_)
          throw std::logic_error("row classword K partition gap");
        if (item.n_begin != n_cursor)
          throw std::logic_error("row classword N partition gap");
        n_cursor = checked_add(n_cursor, item.n_count, "row classword N overflow");
        active_n_begin = item.n_begin;
        active_n_count = item.n_count;
        k_cursor = 0;
      } else if (item.n_count != active_n_count) {
        throw std::logic_error("row classword N segment mismatch");
      }
      if (item.k_begin != k_cursor)
        throw std::logic_error("row classword K partition gap");
      k_cursor = checked_add(k_cursor, item.k_count, "row classword K overflow");
    }
    if (count != 0 && k_cursor != k_tiles_)
      throw std::logic_error("row classword K partition incomplete");
    if (n_cursor != n_tiles_)
      throw std::logic_error("row classword N partition incomplete");
    return count;
  }

  bool same_row_word(u64 a, u64 b) const {
    std::array<RowWordEntry, kMaxClasses> x{}, y{};
    const std::size_t nx = whole_row_word(a, &x), ny = whole_row_word(b, &y);
    if (nx != ny) return false;
    for (std::size_t i = 0; i < nx; ++i) {
      if (x[i].n_begin != y[i].n_begin || x[i].n_count != y[i].n_count ||
          x[i].k_begin != y[i].k_begin || x[i].k_count != y[i].k_count ||
          x[i].cid != y[i].cid) return false;
    }
    return true;
  }

  u128 output_copy_bound(const Snapshot& old, u128 d) const {
    if (d % k_tiles_ != 0) return 0;
    const u128 period_outputs = d / k_tiles_;
    if (period_outputs == 0 || owner_.next_load - old.owner.next_load != d ||
        owner_.retired < old.owner.retired) return 0;
    const u128 q0 = old.owner.next_load / k_tiles_;
    const u128 qcur = owner_.next_load / k_tiles_;
    const u128 expected_qcur = checked_add(q0, period_outputs,
                                           "output-copy q cursor overflow");
    if (qcur != expected_qcur ||
        owner_.retired - old.owner.retired != period_outputs) return 0;
    std::array<u128, 8> live{};
    const std::size_t live_count = refs(&live);
    u128 wave_bound = ~u128{0};
    bool saw_bound = false;
    for (std::size_t i = 0; i < live_count; ++i) {
      const u128 wave = live[i];
      if (wave >= total_waves_) continue;
      const u128 bound = (total_waves_ - 1 - wave) / d;
      wave_bound = std::min(wave_bound, bound);
      saw_bound = true;
    }
    const u128 next_bound = (total_waves_ - owner_.next_load) / d;
    wave_bound = saw_bound ? std::min(wave_bound, next_bound) : next_bound;
    if (wave_bound == 0) return 0;

    if (period_outputs % n_tiles_ == 0) {
      if (q0 >= total_outputs_) return 0;
      const u128 qlast = checked_add(q0, period_outputs - 1,
                                     "output-copy last q overflow");
      if (qlast >= total_outputs_) return 0;
      const auto first = facts_.geometry.output_fact(q0);
      const auto last = facts_.geometry.output_fact(qlast);
      if (first.m != last.m || !same_row_word(first.m_tile, last.m_tile)) return 0;
      for (std::size_t i = 0; i < live_count; ++i) {
        if (live[i] >= total_waves_) continue;
        const u64 row = static_cast<u64>((live[i] / k_tiles_) / n_tiles_);
        if (!same_row_word(first.m_tile, row)) return 0;
      }
      u128 row_bound = wave_bound;
      if (first.m != facts_.input.tm) return 0;
      if (facts_.input.M % facts_.input.tm != 0) {
        const u64 tail_row = m_tiles_ - 1;
        const u128 rows_per_cycle = period_outputs / n_tiles_;
        for (std::size_t i = 0; i < live_count; ++i) {
          if (live[i] >= total_waves_) continue;
          const u64 row = static_cast<u64>((live[i] / k_tiles_) / n_tiles_);
          const u128 bound = row + 1 >= tail_row ? 0
              : static_cast<u128>(tail_row - 1 - row) / rows_per_cycle;
          row_bound = std::min(row_bound, bound);
        }
      }
      return std::min(wave_bound, row_bound);
    }

    if (q0 >= total_outputs_ || qcur >= total_outputs_) return 0;
    const auto source = facts_.geometry.output_fact(q0);
    const u128 run_query_end = checked_add(q0, period_outputs,
                                           "output-copy run end overflow");
    if (facts_.geometry.same_shape_run_end(q0) < run_query_end) return 0;
    if (facts_.geometry.output_fact(qcur).m != source.m ||
        facts_.geometry.output_fact(qcur).n != source.n ||
        facts_.geometry.same_shape_run_end(qcur) < checked_add(
            qcur, period_outputs, "current output-copy run end overflow")) return 0;
    u128 class_bound = wave_bound;
    for (std::size_t i = 0; i < live_count; ++i) {
      if (live[i] >= total_waves_) continue;
      const u128 q = live[i] / k_tiles_;
      const u128 run_end = facts_.geometry.same_shape_run_end(q);
      const u128 bound = run_end > q + 1 ? (run_end - 1 - q) / period_outputs : 0;
      class_bound = std::min(class_bound, bound);
    }
    return std::min(wave_bound, class_bound);
  }

  bool try_identity_skip() {
    if (history_count_ == 0 || owner_.finished) {
      save_snapshot();
      return false;
    }
    for (std::size_t i = history_count_; i-- > 0;) {
      const Snapshot& old = history_[i];
      u128 d = 0;
      if (!topology_return(old, &d)) continue;
      std::array<u128, 8> old_refs{}, new_refs{};
      const std::size_t old_count = snapshot_refs(old, &old_refs);
      const std::size_t new_count = refs(&new_refs);
      u128 count = 0;
      bool output_period = false;
      if (d < k_tiles_) {
        bool valid = true;
        u64 max_k = 0;
        for (std::size_t j = 0; j < old_count + new_count; ++j) {
          const u128 wave = j < old_count ? old_refs[j] : new_refs[j - old_count];
          if (wave / k_tiles_ != snapshot_anchor(old) / k_tiles_ ||
              wave % k_tiles_ == 0 || wave % k_tiles_ >= k_tiles_ - 1) {
            valid = false;
            break;
          }
          max_k = std::max(max_k, static_cast<u64>(wave % k_tiles_));
        }
        if (!valid || k_tiles_ <= 2 + max_k) continue;
        count = (k_tiles_ - 2 - max_k) / d;
      } else {
        count = output_copy_bound(old, d);
        output_period = true;
      }
      if (count == 0) continue;
      const Ledger unit = ledger_delta(ledger_, old.ledger);
      if (stop_) {
        if (r_compare(unit.elapsed, Rational{}) <= 0)
          throw std::logic_error("identity return has nonpositive elapsed time");
        const Rational remaining = r_sub(*stop_, ledger_.elapsed);
        if (r_compare(remaining, Rational{}) < 0) continue;
        const Rational ratio = r_div(remaining, unit.elapsed);
        const u128 limit = floor_u128(ratio);
        count = std::min(count, limit);
      }
      if (count == 0) continue;
      ledger_add_repeat(&ledger_, unit, count);
      const u128 shift = checked_mul(d, count, "identity wave shift overflow");
      for (auto& active : owner_.active)
        if (active.present) active.wave = checked_add(active.wave, shift,
                                                       "active wave shift overflow");
      for (std::size_t j = 0; j < owner_.loaded_count; ++j)
        owner_.loaded[j] = checked_add(owner_.loaded[j], shift,
                                       "loaded wave shift overflow");
      for (std::size_t j = 0; j < owner_.computed_count; ++j)
        owner_.computed[j] = checked_add(owner_.computed[j], shift,
                                         "computed wave shift overflow");
      owner_.next_load = checked_add(owner_.next_load, shift,
                                     "next load skip overflow");
      const u128 output_shift = output_period ? (d / k_tiles_) * count : shift / k_tiles_;
      owner_.retired = checked_add(owner_.retired, output_shift,
                                   "retired output skip overflow");
      skipped_waves_ = checked_add(skipped_waves_, shift, "skipped wave count overflow");
      if (output_period)
        skipped_outputs_ = checked_add(skipped_outputs_, output_shift,
                                       "skipped output count overflow");
      save_snapshot();
      return true;
    }
    save_snapshot();
    return false;
  }

  static u128 floor_u128(const Rational& value) {
    if (value.negative()) return 0;
    const u128 maximum = ~u128{0};
    const Rational max_value = r_u128(maximum);
    if (r_compare(value, max_value) > 0)
      throw std::overflow_error("stop-clamped identity count exceeds checked u128");
    if (r_compare(value, max_value) == 0) return maximum;
    u128 low = 0, high = maximum;
    while (low < high) {
      const u128 distance = high - low;
      const u128 middle = low + distance / 2 + distance % 2;
      if (r_compare(value, r_u128(middle)) >= 0) low = middle;
      else high = middle - 1;
    }
    return low;
  }

  void save_snapshot() {
    if (history_count_ >= history_.size()) return;
    history_[history_count_++] = {owner_, ledger_};
  }

  struct RateFact { std::size_t stage; u128 wave; Rational progress; };

  std::array<Rational, kStages> allocate_rates(
      std::array<RateFact, kStages>* active, std::size_t active_count) const {
    std::array<Rational, kStages> rates{};
    if (active_count == 0) throw std::logic_error("allocator has no active stages");
    std::array<const ActivityFact*, kStages> fact{};
    Rational owner_q{};
    for (std::size_t i = 0; i < active_count; ++i) {
      fact[i] = &facts_.activity((*active)[i].stage, (*active)[i].wave);
      owner_q = r_add(owner_q, fact[i]->canonical_q);
    }
    if (r_compare(owner_q, Rational{}) <= 0)
      throw std::logic_error("owner canonical Q is not positive");
    std::array<bool, kStages> free{};
    for (std::size_t i = 0; i < active_count; ++i) free[i] = true;
    for (std::size_t tier = 0; tier < active_count; ++tier) {
      if (std::none_of(free.begin(), free.begin() + active_count,
                       [](bool value) { return value; }))
        break;
      std::array<Rational, kRows + kStages> bounds{};
      std::array<bool, kRows + kStages> has_bound{};
      for (std::size_t row = 0; row < kRows; ++row) {
        Rational used{}, slope{};
        for (std::size_t i = 0; i < active_count; ++i) {
          used = r_add(used, r_mul(fact[i]->work[row], rates[i]));
          if (free[i]) slope = r_add(slope,
                                     r_div(fact[i]->work[row], owner_q));
        }
        if (!slope.zero()) {
          bounds[row] = r_div(r_sub(capacity_[row], used), slope);
          has_bound[row] = true;
        }
      }
      for (std::size_t i = 0; i < active_count; ++i) {
        if (!free[i] || fact[i]->private_time.zero()) continue;
        const Rational inv = r_div(r_integer("1"), fact[i]->private_time);
        bounds[kRows + i] = r_mul(r_sub(inv, rates[i]), owner_q);
        has_bound[kRows + i] = true;
      }
      bool have = false;
      Rational step{};
      for (std::size_t i = 0; i < bounds.size(); ++i) {
        if (!has_bound[i]) continue;
        if (!have || r_compare(bounds[i], step) < 0) {
          step = bounds[i]; have = true;
        }
      }
      if (!have) throw std::logic_error("unbounded progressive-fill tier");
      if (r_compare(step, Rational{}) < 0)
        throw std::logic_error("negative progressive-fill step");
      const Rational increment = r_div(step, owner_q);
      for (std::size_t i = 0; i < active_count; ++i)
        if (free[i]) rates[i] = r_add(rates[i], increment);
      std::array<bool, kStages> frozen{};
      for (std::size_t row = 0; row < kRows; ++row) {
        if (!has_bound[row] || !bounds[row].equal(step)) continue;
        for (std::size_t i = 0; i < active_count; ++i)
          if (free[i] && !fact[i]->work[row].zero()) frozen[i] = true;
      }
      for (std::size_t i = 0; i < active_count; ++i)
        if (has_bound[kRows + i] && bounds[kRows + i].equal(step)) frozen[i] = true;
      bool any = false;
      for (std::size_t i = 0; i < active_count; ++i) {
        if (free[i] && frozen[i]) { free[i] = false; any = true; }
      }
      if (!any) throw std::logic_error("progressive-fill tier froze no limiter");
    }
    for (std::size_t row = 0; row < kRows; ++row) {
      Rational used{};
      for (std::size_t i = 0; i < active_count; ++i)
        used = r_add(used, r_mul(fact[i]->work[row], rates[i]));
      if (r_compare(used, capacity_[row]) > 0)
        throw std::logic_error("allocation exceeds a 13-domain capacity");
    }
    return rates;
  }

  void event() {
    if (events_ >= kGlobalEventCap)
      throw std::logic_error("global expanded-event cap reached");
    settle();
    if (owner_.finished) return;
    std::array<RateFact, kStages> active{};
    std::size_t active_count = 0;
    for (std::size_t s = 0; s < kStages; ++s)
      if (owner_.active[s].present)
        active[active_count++] = {s, owner_.active[s].wave, owner_.active[s].progress};
    peak_active_ = std::max<u64>(peak_active_, active_count);
    if (active_count == 0 || active_count > kStages)
      throw std::logic_error("invalid active L/C/S stage count");
    const auto rates = allocate_rates(&active, active_count);
    std::array<Rational, kStages> deadlines{};
    std::array<bool, kStages> has_deadline{};
    Rational delta{};
    bool have_delta = false;
    for (std::size_t i = 0; i < active_count; ++i) {
      if (rates[i].zero()) continue;
      const Rational remain = r_sub(r_integer("1"), active[i].progress);
      deadlines[i] = r_div(remain, rates[i]);
      has_deadline[i] = true;
      if (!have_delta || r_compare(deadlines[i], delta) < 0) {
        delta = deadlines[i]; have_delta = true;
      }
    }
    if (!have_delta) throw std::logic_error("deadlock: all active rates are zero");
    if (stop_) {
      if (r_compare(*stop_, ledger_.elapsed) <= 0)
        throw std::logic_error("event attempted at or after external stop");
      const Rational until_stop = r_sub(*stop_, ledger_.elapsed);
      if (r_compare(until_stop, delta) < 0) delta = until_stop;
    }
    if (r_compare(delta, Rational{}) <= 0)
      throw std::logic_error("nonpositive lifecycle event interval");
    std::array<bool, kStages> selected{};
    for (std::size_t i = 0; i < active_count; ++i)
      selected[i] = has_deadline[i] && deadlines[i].equal(delta);

    ledger_.elapsed = r_add(ledger_.elapsed, delta);
    std::array<bool, kRows> busy{};
    for (std::size_t i = 0; i < active_count; ++i) {
      const ActivityFact& fact = facts_.activity(active[i].stage, active[i].wave);
      const Rational portion = r_mul(delta, rates[i]);
      owner_.active[active[i].stage].progress = r_add(active[i].progress, portion);
      for (std::size_t row = 0; row < kRows; ++row) {
        ledger_.service[row] = r_add(ledger_.service[row],
                                    r_mul(portion, fact.work[row]));
        if (!rates[i].zero() && !fact.work[row].zero()) busy[row] = true;
      }
      for (std::size_t row = 0; row < 2; ++row)
        for (std::size_t dir = 0; dir < 2; ++dir)
          ledger_.hbm_payload[row][dir] = r_add(
              ledger_.hbm_payload[row][dir], r_mul(portion, fact.hbm[row][dir]));
      ledger_.int4_ops = r_add(ledger_.int4_ops, r_mul(portion, fact.int4_ops));
      ledger_.vec_eqops = r_add(ledger_.vec_eqops, r_mul(portion, fact.vec_eqops));
    }
    for (std::size_t row = 0; row < kRows; ++row)
      if (busy[row]) ledger_.busy[row] = r_add(ledger_.busy[row], delta);
    if (busy[0]) ledger_.compute_busy = r_add(ledger_.compute_busy, delta);
    if (busy[4]) ledger_.hbm_row_busy[0] = r_add(ledger_.hbm_row_busy[0], delta);
    if (busy[5]) ledger_.hbm_row_busy[1] = r_add(ledger_.hbm_row_busy[1], delta);
    if (busy[4] || busy[5]) ledger_.hbm_busy = r_add(ledger_.hbm_busy, delta);

    for (std::size_t i = 0; i < active_count; ++i) {
      if (!selected[i]) continue;
      const std::size_t stage = active[i].stage;
      const u128 wave = active[i].wave;
      owner_.active[stage] = Active{};
      if (stage == 0) {
        if (owner_.loaded_count >= kQueueSlots)
          throw std::logic_error("input readiness queue overflow");
        owner_.loaded[owner_.loaded_count++] = wave;
      } else if (stage == 1) {
        if (facts_.is_final(wave)) {
          if (owner_.computed_count >= kQueueSlots)
            throw std::logic_error("output readiness queue overflow");
          owner_.computed[owner_.computed_count++] = wave;
        }
      } else {
        owner_.retired = checked_add(owner_.retired, 1,
                                     "retired output count overflow");
      }
    }
    ++events_;
    settle();
  }

  void validate_complete() const {
    if (owner_.retired != total_outputs_ || !no_live_work() ||
        owner_.next_load != total_waves_)
      throw std::logic_error("lifecycle completed with residual work");
    Ledger expected;
    for (std::size_t i = 0; i < facts_.geometry.class_count(); ++i) {
      const auto& cls = facts_.geometry.access_class(i);
      const u128 repetitions = cls.repeat_count;
      const Rational reps = r_u128(repetitions);
      for (std::size_t stage = 0; stage < kStages; ++stage) {
        if (stage == 2 && !cls.final_k) continue;
        const auto& fact = facts_.classes[i].stage[stage];
        for (std::size_t row = 0; row < kRows; ++row)
          expected.service[row] = r_add(expected.service[row],
                                         r_mul(reps, fact.work[row]));
        for (std::size_t row = 0; row < 2; ++row)
          for (std::size_t dir = 0; dir < 2; ++dir)
            expected.hbm_payload[row][dir] = r_add(expected.hbm_payload[row][dir],
                r_mul(reps, fact.hbm[row][dir]));
        expected.int4_ops = r_add(expected.int4_ops, r_mul(reps, fact.int4_ops));
        expected.vec_eqops = r_add(expected.vec_eqops, r_mul(reps, fact.vec_eqops));
      }
    }
    for (std::size_t row = 0; row < kRows; ++row)
      if (!expected.service[row].equal(ledger_.service[row]))
        throw std::logic_error("completed service ledger differs from actual class facts");
    for (std::size_t row = 0; row < 2; ++row)
      for (std::size_t dir = 0; dir < 2; ++dir)
        if (!expected.hbm_payload[row][dir].equal(ledger_.hbm_payload[row][dir]))
          throw std::logic_error("completed HBM payload differs from actual class facts");
    if (!expected.int4_ops.equal(ledger_.int4_ops) ||
        !expected.vec_eqops.equal(ledger_.vec_eqops))
      throw std::logic_error("completed typed operation ledger differs from actual facts");
  }
};

}  // namespace solo13
