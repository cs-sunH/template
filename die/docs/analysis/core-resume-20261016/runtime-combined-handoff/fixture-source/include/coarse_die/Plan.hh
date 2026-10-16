#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <json/json.hpp>
#include "coarse_die/Hardware.hh"

namespace coarse_die {
constexpr std::size_t resource_count = 13;
enum ResourceId : std::size_t { COMPUTE_GATE, SRAM_READ, SRAM_WRITE, LOCAL,
    HBM0, HBM1, INJECT0, INJECT1, XPOS, XNEG, YPOS, YNEG, HOP };
enum class Role : unsigned { Load, Compute, Store, Endpoint };
struct CreditPoint {
    double progress = 0;
    std::array<double,2> cumulative{};
};
struct Activity {
    std::uint64_t id = 0;
    Role role = Role::Compute;
    std::array<double,resource_count> work{};
    std::array<std::array<std::uint64_t,2>,2> hbm_bytes{};
    std::uint64_t int4_ops = 0, vec_eqops = 0;
    // Weighted equivalent work can be fractional. Keep the whole part exact
    // for existing integer Plans; a nonzero remainder uses Plan schema v3.
    long double vec_eqops_fraction = 0;
    double canonical_q = 1, private_time_s = 0;
    // Fractions of each activity's [0,1] progress alter buffer occupancy.
    // Zero means that the activity does not touch this buffer.
    std::array<double,2> credit_delta{};
    // Optional piecewise-linear cumulative credit. Empty preserves the legacy
    // linear credit_delta * progress description.
    std::vector<CreditPoint> credit_curve;
};

std::array<double,2> credit_at(const Activity&, double progress);
// At an internal curve point, returns the slope of the segment to its right.
std::array<double,2> credit_slope(const Activity&, double progress);
// Returns the first curve point strictly greater than progress, or 1.
double next_credit_breakpoint(const Activity&, double progress);

inline long double planned_vec_eqops(const Activity& activity) {
    const long double whole = static_cast<long double>(activity.vec_eqops);
    const long double result = whole + activity.vec_eqops_fraction;
    if (activity.vec_eqops_fraction != 0 &&
        result - whole != activity.vec_eqops_fraction)
        throw std::overflow_error("overflow: fractional Vec work is lost in runtime scalar precision");
    return result;
}

// Canonical exact dyadic representation, independent of decimal formatting.
// A nonzero fraction is an odd significand times 2^exponent, in (0,1).
nlohmann::json equivalent_fraction_to_json(long double fraction);
long double equivalent_fraction_from_json(const nlohmann::json& fraction);

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

// Count every independent activity's affine pieces, including its terminal
// switch; an empty epoch contributes one closure. Different activities do not
// share a budget entry merely because their progress endpoints are equal.
// A nonzero initial ready delay adds one independent gate/switch.
std::size_t service_description_count(const Plan&);

inline bool has_credit_curves(const Plan& plan) {
    for(const auto& epoch:plan.epochs)
        for(const auto& activity:epoch.activities)
            if(!activity.credit_curve.empty())return true;
    return false;
}

// Capacity consumers must not use json::get<uint64_t>() directly: that cast
// accepts negative or floating-point values and can silently change the byte
// count. Workspace is the compiler's exact incremental HBM capacity, separate
// from weights/KV and from transfer payload. Missing metadata is an error.
inline std::uint64_t hbm_workspace_peak_bytes(const Plan& plan) {
    if (!plan.summary.is_object())
        throw std::invalid_argument("invalid_workspace: Plan summary must be an object");
    const auto value = plan.summary.find("hbm_workspace_peak_bytes");
    if (value == plan.summary.end())
        throw std::invalid_argument("invalid_workspace: hbm_workspace_peak_bytes is required");
    if (value->is_number_unsigned()) return value->get<std::uint64_t>();
    if (value->is_number_integer()) {
        const auto bytes = value->get<std::int64_t>();
        if (bytes >= 0) return static_cast<std::uint64_t>(bytes);
    }
    throw std::invalid_argument("invalid_workspace: hbm_workspace_peak_bytes must be an exact uint64 integer");
}
inline bool has_fractional_work(const Plan& plan) {
    for (const auto& epoch : plan.epochs)
        for (const auto& activity : epoch.activities)
            if (activity.vec_eqops_fraction != 0) return true;
    return false;
}
std::shared_ptr<const Plan> compile(const Hardware&, const nlohmann::json& task);
// Compile-only capacity certificate for one task. This enumerates the same
// normalized mapping candidates as compile(), but skips compile()'s outer
// latency ranking and does not submit work to a caller's Runtime, publish a
// Plan handle, or mutate an ABI model cache. Mapping enumeration can still use
// its existing internal latency estimates for component-candidate selection.
struct WorkspaceEnvelope {
    std::uint64_t candidate_count = 0;
    std::uint64_t max_hbm_workspace_peak_bytes = 0;
    std::string hardware_fingerprint;
    std::string normalized_task_fingerprint;
    std::string op_kind;
    std::string phase;
    nlohmann::json normalized_task;
};
WorkspaceEnvelope compile_workspace_envelope(const Hardware&,
                                              const nlohmann::json& task);
// Candidate-independent capacity upper bound for every template permitted by
// the normalized task. This intentionally omits SRAM feasibility filtering and
// does not prove that the task has an executable mapping.
struct WorkspaceCapacityBound {
    std::uint64_t hbm_workspace_upper_bound_bytes = 0;
    std::string hardware_fingerprint;
    std::string normalized_task_fingerprint;
    std::string op_kind;
    std::string phase;
    nlohmann::json normalized_task;
};
WorkspaceCapacityBound compile_workspace_capacity_bound(
    const Hardware&, const nlohmann::json& task);
nlohmann::json estimate_alone(const Hardware&, const Plan&);
std::string stable_hash(const std::string& text);
nlohmann::json plan_to_json(const Plan&);
std::shared_ptr<const Plan> plan_from_json(const nlohmann::json&);

namespace detail {
void validate_credit_curve(const Activity&);
// Checked-u128 lifetime formula shared by mapping workspace assignment and
// the candidate-independent compiler bound.
std::uint64_t workspace_capacity_upper_bound(
    const Hardware&, const nlohmann::json& normalized_task);
// Run the structural checks used by plan_to_json() without allocating a JSON
// payload or computing its canonical hash. Compiler-only unsealed candidates
// use this before their Runtime trial; public Plans still use plan_to_json().
void validate_plan_structure(const Plan& plan);
// Candidate-ranking driver for an already compiled, canonically fingerprinted
// Plan. Runtime::submit still validates hardware identity and physical work;
// this path avoids building the public standalone estimate report.
std::uint64_t estimate_latency_ns(const Hardware&,
                                  std::shared_ptr<const Plan> plan);
std::uint64_t estimate_latency_ns(std::shared_ptr<const Hardware> hardware,
                                  std::shared_ptr<const Plan> plan);
// Compiler-only trial of an unsealed candidate. Uses the same Runtime and
// physical validation, returns only a latency, and publishes no Plan/handle
// or snapshot. The selected candidate must be sealed before it is returned.
std::uint64_t estimate_candidate_latency_ns(
    std::shared_ptr<const Hardware> hardware,
    std::shared_ptr<const Plan> plan);
}
}
