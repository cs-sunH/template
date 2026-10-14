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

}  // namespace

int main(int argc, char** argv) {
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

    Kernel zero_capacity(dense);
    std::array<Rational, kRows> zero{};
    const auto zero_status = zero_capacity.resume_until(nullptr, &zero);
    results["zero_capacity"] = result_for(zero_capacity, zero_status);
    if (zero_status != Kernel::Status::FailedClosed)
      throw std::runtime_error("zero capacities did not fail closed");

    Kernel clamped(large_k);
    const Rational huge_stop = r_integer(pow2_decimal(200));
    const auto clamp_status = clamped.run_until(&huge_stop);
    results["stop_clamp_u128_overflow"] = result_for(clamped, clamp_status);
    if (clamp_status != Kernel::Status::FailedClosed ||
        clamped.error().find("stop-clamped identity count exceeds checked u128") == std::string::npos)
      throw std::runtime_error("oversized stop clamp status=" +
          std::to_string(static_cast<int>(clamp_status)) + " reason=" + clamped.error());

    Kernel event_cap(dense, false);
    const auto cap_status = event_cap.run_until();
    results["unpowered_event_cap"] = result_for(event_cap, cap_status);
    if (cap_status != Kernel::Status::FailedClosed || event_cap.events() != kGlobalEventCap)
      throw std::runtime_error("unpowered event cap did not fail exactly at 256");

    results["sanitizer_scope"] = "scratch kernel overflow, zero capacity, stop-clamp overflow, and global event boundary smoke";
    results["status"] = "passed";
    std::cout << results.dump(2) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "sanitizer smoke failed: " << error.what() << '\n';
    return 2;
  }
}
