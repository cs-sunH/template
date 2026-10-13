#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <json/json.hpp>
#include "coarse_die/Hardware.hh"

namespace coarse_die {
constexpr std::size_t resource_count = 13;
enum ResourceId : std::size_t { COMPUTE_GATE, SRAM_READ, SRAM_WRITE, LOCAL,
    HBM0, HBM1, INJECT0, INJECT1, XPOS, XNEG, YPOS, YNEG, HOP };
enum class Role : unsigned { Load, Compute, Store, Endpoint };
struct Activity {
    std::uint64_t id = 0;
    Role role = Role::Compute;
    std::array<double,resource_count> work{};
    std::array<std::array<std::uint64_t,2>,2> hbm_bytes{};
    std::uint64_t int4_ops = 0, vec_eqops = 0;
    double canonical_q = 1, private_time_s = 0;
    // Fractions of each activity's [0,1] progress alter buffer occupancy.
    // Zero means that the activity does not touch this buffer.
    std::array<double,2> credit_delta{};
};

// The fairness normalization is a canonical logical quantity, not the sum of
// physical path work (which may include D/eta or multiple routed hops).
// Callers that have a pure NoC payload should pass that logical byte count.
inline double canonical_basis(const Activity& activity,
                              double logical_noc_payload = 0.0) {
    long double hbm_payload = 0.0L;
    for (const auto& row : activity.hbm_bytes)
        for (std::uint64_t bytes : row) hbm_payload += bytes;
    if (hbm_payload > 0.0L) return static_cast<double>(hbm_payload);
    if (activity.work[LOCAL] > 0.0) return activity.work[LOCAL];
    const double sram = activity.work[SRAM_READ] + activity.work[SRAM_WRITE];
    if (sram > 0.0) return sram;
    if (logical_noc_payload > 0.0) return logical_noc_payload;
    if (activity.work[COMPUTE_GATE] > 0.0) return 1.0;
    return 1.0;
}

struct Epoch {
    std::string name;
    std::vector<Activity> activities;
    std::array<double,2> credit_initial{}, credit_capacity{};
    std::uint64_t repeat_count = 1;
};
struct Plan {
    nlohmann::json task, summary;
    std::string fingerprint, hardware_fingerprint;
    std::vector<Epoch> epochs;
    std::uint64_t launch_ns = 0;
};
std::shared_ptr<const Plan> compile(const Hardware&, const nlohmann::json& task);
nlohmann::json estimate_alone(const Hardware&, const Plan&);
std::string stable_hash(const std::string& text);
nlohmann::json plan_to_json(const Plan&);
std::shared_ptr<const Plan> plan_from_json(const nlohmann::json&);
}
