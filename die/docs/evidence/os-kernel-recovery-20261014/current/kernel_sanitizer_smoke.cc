#include "solo13_kernel.hh"

#include <fstream>
#include <iostream>
#include <string>

namespace {

using namespace solo13;

json facts(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open facts: " + path);
  return json::parse(stream);
}

json result_for(const Kernel& kernel, Kernel::Status status) {
  const auto status_name = [](Kernel::Status value) {
    switch (value) {
      case Kernel::Status::Passed: return "passed";
      case Kernel::Status::Paused: return "paused";
      case Kernel::Status::FailedClosed: return "failed_closed";
    }
    return "unknown";
  };
  return {{"status", status_name(status)}, {"events", kernel.events()},
          {"error", kernel.error()},
          {"skipped_waves", u128_string(kernel.skipped_waves())}};
}

bool ledger_equal(const Ledger& left, const Ledger& right) {
  if (!left.elapsed.equal(right.elapsed) ||
      !left.int4_ops.equal(right.int4_ops) ||
      !left.vec_eqops.equal(right.vec_eqops) ||
      !left.compute_busy.equal(right.compute_busy) ||
      !left.hbm_busy.equal(right.hbm_busy)) return false;
  for (std::size_t row = 0; row < kRows; ++row)
    if (!left.service[row].equal(right.service[row]) ||
        !left.busy[row].equal(right.busy[row])) return false;
  for (std::size_t row = 0; row < 2; ++row) {
    if (!left.hbm_row_busy[row].equal(right.hbm_row_busy[row])) return false;
    for (std::size_t direction = 0; direction < 2; ++direction)
      if (!left.hbm_payload[row][direction].equal(
              right.hbm_payload[row][direction])) return false;
  }
  return true;
}

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main(int argc, char** argv) {
  json results = json::object();
  try {
    if (argc != 3) throw std::invalid_argument("usage: sanitizer_smoke DENSE_FACTS LARGE_K_FACTS");
    json dense = facts(argv[1]);
    json large_k = facts(argv[2]);
    json results = json::object();

    bool overflow_detected = false;
    try {
      geom::Input huge;
      huge.M = huge.N = huge.K = std::numeric_limits<u64>::max();
      huge.tm = huge.tn = huge.tk = huge.core_count = 1;
      geom::GemmOSGeometry impossible(huge);
      (void)impossible.consumption_wave_count();
    } catch (const std::overflow_error& error) {
      overflow_detected = true;
      results["geometry_u128_overflow"] = {{"status", "failed_closed"},
                                           {"reason", error.what()}};
    }
    if (!overflow_detected) throw std::runtime_error("geometry overflow was not rejected");

    constexpr const char* kMax256 =
        "115792089237316195423570985008687907853269984665640564039457584007913129639935";
    const std::string max_u128 = "340282366920938463463374607431768211455";
    const std::string one_past_u128 = "340282366920938463463374607431768211456";
    Rational state_max;
    std::string fixedwidth_error;
    require(Rational::from_integer(kMax256, &state_max, &fixedwidth_error) &&
                state_max.numerator_bits() == 256 && state_max.denominator_bits() == 1,
            "2^256-1 did not fit the fixed-width state slot");
    results["fixedwidth_256bit_numerator_max"] = {
        {"status", "passed"}, {"numerator_bits", state_max.numerator_bits()},
        {"denominator_bits", state_max.denominator_bits()}};
    bool numerator_overflow = false;
    try {
      (void)r_integer(pow2_decimal(256));
    } catch (const std::overflow_error& error) {
      numerator_overflow = std::string(error.what()).find(
          "normalized rational exceeds 256-bit state slot") != std::string::npos;
      fixedwidth_error = error.what();
    }
    require(numerator_overflow, "2^256 numerator did not hit the 256-bit state boundary");
    results["fixedwidth_256bit_numerator_overflow"] = {
        {"status", "failed_closed"}, {"reason", fixedwidth_error}};
    bool denominator_overflow = false;
    try {
      (void)r_fraction("1/" + pow2_decimal(256));
    } catch (const std::overflow_error& error) {
      denominator_overflow = std::string(error.what()).find(
          "normalized rational exceeds 256-bit state slot") != std::string::npos;
      fixedwidth_error = error.what();
    }
    require(denominator_overflow, "1/2^256 denominator did not hit the 256-bit state boundary");
    results["fixedwidth_256bit_denominator_overflow"] = {
        {"status", "failed_closed"}, {"reason", fixedwidth_error}};

    Rational repeated;
    std::string repeat_error;
    require(Rational::multiply_repeat(r_integer("1"), max_u128,
                                      &repeated, &repeat_error) &&
                repeated.equal(r_integer(max_u128)),
            "maximum checked-u128 repeat was rejected or changed");
    results["fixedwidth_repeat_u128_max"] = {
        {"status", "passed"}, {"value", max_u128}};
    require(!Rational::multiply_repeat(r_integer("1"), one_past_u128,
                                      &repeated, &repeat_error) &&
                repeat_error == "repeat count exceeds checked u128 range",
            "repeat count above u128 did not fail at its checked boundary");
    results["fixedwidth_repeat_u128_overflow"] = {
        {"status", "failed_closed"}, {"reason", repeat_error}};
    Kernel zero_capacity(dense);
    std::array<Rational, kRows> zero{};
    const auto zero_status = zero_capacity.resume_until(nullptr, &zero);
    results["zero_capacity"] = result_for(zero_capacity, zero_status);
    if (zero_status != Kernel::Status::FailedClosed)
      throw std::runtime_error("zero capacities did not fail closed");

    Kernel clamped(large_k);
    const Rational huge_stop = r_integer(one_past_u128);
    const auto clamp_status = clamped.run_until(&huge_stop);
    results["stop_clamp_u128_overflow"] = result_for(clamped, clamp_status);
    if (clamp_status != Kernel::Status::FailedClosed ||
        clamped.error().find("stop-clamped identity count exceeds checked u128") == std::string::npos)
      throw std::runtime_error("oversized stop clamp status=" +
          std::to_string(static_cast<int>(clamp_status)) + " reason=" + clamped.error());
    results["fixedwidth_floor_u128_overflow"] = {
        {"status", "failed_closed"}, {"reason", clamped.error()},
        {"stop", huge_stop.to_string()}};

    Kernel event_cap(dense, false);
    const auto cap_status = event_cap.run_until();
    results["unpowered_event_cap"] = result_for(event_cap, cap_status);
    if (cap_status != Kernel::Status::FailedClosed || event_cap.events() != kGlobalEventCap)
      throw std::runtime_error("unpowered event cap did not fail exactly at 256");

    Kernel reset_probe(dense);
    const Rational reset_stop = r_fraction("17/1000000000");
    const auto reset_pause = reset_probe.run_until(&reset_stop);
    require(reset_pause == Kernel::Status::Paused,
            "reset probe did not reach its expected external-stop pause");
    auto changed = reset_probe.capacities();
    const Rational slower = r_fraction("7/8");
    changed[4] = r_mul(changed[4], slower);
    changed[5] = r_mul(changed[5], slower);
    require(reset_probe.resume_until(nullptr, &changed) == Kernel::Status::Passed,
            "reset probe did not finish after a valid capacity update");
    reset_probe.reset();
    bool reset_fields_ok = reset_probe.events() == 0 &&
        reset_probe.skipped_waves() == 0 && reset_probe.skipped_outputs() == 0 &&
        reset_probe.history_count() == 0 && reset_probe.capacity_invalidations() == 0 &&
        reset_probe.peak_active() == 0 && reset_probe.peak_input() == 0 &&
        reset_probe.peak_output() == 0 &&
        reset_probe.nonfinal_compute_at_full_output() == 0 &&
        reset_probe.final_compute_blocked_on_full_output() == 0 &&
        reset_probe.error().empty() && !reset_probe.owner().finished &&
        reset_probe.owner().next_load == 0 && reset_probe.owner().retired == 0;
    for (std::size_t row = 0; row < kRows; ++row) {
      reset_fields_ok = reset_fields_ok &&
          reset_probe.capacities()[row].equal(reset_probe.facts().capacities[row]) &&
          reset_probe.ledger().service[row].zero() && reset_probe.ledger().busy[row].zero();
    }
    require(reset_fields_ok, "reset did not restore empty state and input capacities");
    Kernel fresh(dense);
    require(fresh.run_until() == Kernel::Status::Passed &&
                reset_probe.run_until() == Kernel::Status::Passed &&
                reset_probe.events() == fresh.events() &&
                reset_probe.skipped_waves() == fresh.skipped_waves() &&
                reset_probe.skipped_outputs() == fresh.skipped_outputs() &&
                ledger_equal(reset_probe.ledger(), fresh.ledger()),
            "reset full run differs from a fresh kernel run");
    results["reset_after_capacity_update"] = {
        {"status", "passed"}, {"events_after_reset_run", reset_probe.events()},
        {"capacity_invalidations_after_reset_run", reset_probe.capacity_invalidations()}};

    results["sanitizer_scope"] = "scratch reset, exact 256-bit rational and checked-u128 boundaries, zero capacity, stop-clamp overflow, and global event boundary smoke";
    results["status"] = "passed";
    std::cout << results.dump(2) << '\n';
    return 0;
  } catch (const std::exception& error) {
    results["status"] = "failed";
    results["failure"] = error.what();
    std::cout << results.dump(2) << '\n';
    std::cerr << "sanitizer smoke failed: " << error.what() << '\n';
    return 2;
  }
}
