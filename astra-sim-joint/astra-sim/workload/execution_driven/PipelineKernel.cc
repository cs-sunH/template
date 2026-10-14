/******************************************************************************
Bounded integer window algebra for the die timer-window approximation, plus
the PipelineKernelContract implementation (D4: migrated from the former
PipelineDescriptor.cc when the remote-read tile descriptors were retired).
******************************************************************************/

#include "astra-sim/workload/execution_driven/PipelineKernel.hh"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace AstraSim {
namespace ExecutionDriven {
namespace {

[[noreturn]] void fail(const char* message) {
    throw std::invalid_argument(std::string("pipeline kernel: ") + message);
}

[[noreturn]] void fail(const std::string& path, const std::string& reason) {
    throw std::invalid_argument("pipeline kernel " + path + ": " + reason);
}

bool valid_window_limit(uint32_t limit) {
    return limit == 1 || limit == 2 || limit == 4 || limit == 8;
}

uint64_t add_counter(uint64_t value, uint64_t delta, const char* label) {
    if (delta > std::numeric_limits<uint64_t>::max() - value) {
        throw std::overflow_error(std::string("pipeline window: ") + label +
                                  " counter overflow");
    }
    return value + delta;
}

uint32_t greatest_window_level(uint32_t limit) {
    for (const uint32_t level : {8U, 4U, 2U, 1U}) {
        if (level <= limit) return level;
    }
    return 1;
}

void require_exact_keys(const nlohmann::json& value, const std::string& path,
                        const std::set<std::string>& required) {
    if (!value.is_object()) fail(path, "expected object");
    for (const auto& key : required) {
        if (value.find(key) == value.end()) {
            fail(path, "missing key '" + key + "'");
        }
    }
    for (auto it = value.begin(); it != value.end(); ++it) {
        if (required.count(it.key()) == 0) {
            fail(path, "unknown key '" + it.key() + "'");
        }
    }
}

uint64_t get_u64(const nlohmann::json& value, const std::string& path) {
    if (value.is_number_unsigned()) return value.get<uint64_t>();
    if (!value.is_number_integer()) fail(path, "expected unsigned integer");
    const int64_t signed_value = value.get<int64_t>();
    if (signed_value < 0) fail(path, "must be nonnegative");
    return static_cast<uint64_t>(signed_value);
}

uint32_t get_u32(const nlohmann::json& value, const std::string& path) {
    const uint64_t parsed = get_u64(value, path);
    if (parsed > std::numeric_limits<uint32_t>::max()) {
        fail(path, "exceeds uint32");
    }
    return static_cast<uint32_t>(parsed);
}

std::string get_nonempty_string(const nlohmann::json& value,
                                const std::string& path) {
    if (!value.is_string()) fail(path, "expected string");
    auto parsed = value.get<std::string>();
    if (parsed.empty()) fail(path, "must not be empty");
    return parsed;
}

}  // namespace

PipelineKernelContract::PipelineKernelContract(
    uint32_t projection_abi_version, uint32_t descriptor_version,
    uint32_t pipeline_protocol_version, std::string model_version,
    std::string timer_whitelist_version, uint32_t window_tile_limit,
    uint32_t window_local_error_ppm, uint32_t window_refine_margin_ppm,
    std::string prediction_config_sha256)
    : projection_abi_version_(projection_abi_version),
      descriptor_version_(descriptor_version),
      pipeline_protocol_version_(pipeline_protocol_version),
      model_version_(std::move(model_version)),
      timer_whitelist_version_(std::move(timer_whitelist_version)),
      window_tile_limit_(window_tile_limit),
      window_local_error_ppm_(window_local_error_ppm),
      window_refine_margin_ppm_(window_refine_margin_ppm),
      prediction_config_sha256_(std::move(prediction_config_sha256)) {}

std::shared_ptr<const PipelineKernelContract> PipelineKernelContract::parse(
    const nlohmann::json& wire) {
    const std::string path = "kernel_contract";
    require_exact_keys(wire, path,
                       {"projection_abi_version", "descriptor_version",
                        "pipeline_protocol_version", "model_version",
                        "timer_whitelist_version", "window_tile_limit",
                        "window_local_error_ppm", "window_refine_margin_ppm",
                        "prediction_config_sha256"});
    auto result = std::shared_ptr<const PipelineKernelContract>(
        new PipelineKernelContract(
            get_u32(wire.at("projection_abi_version"), path + ".projection_abi_version"),
            get_u32(wire.at("descriptor_version"), path + ".descriptor_version"),
            get_u32(wire.at("pipeline_protocol_version"), path + ".pipeline_protocol_version"),
            get_nonempty_string(wire.at("model_version"), path + ".model_version"),
            get_nonempty_string(wire.at("timer_whitelist_version"),
                                path + ".timer_whitelist_version"),
            get_u32(wire.at("window_tile_limit"), path + ".window_tile_limit"),
            get_u32(wire.at("window_local_error_ppm"), path + ".window_local_error_ppm"),
            get_u32(wire.at("window_refine_margin_ppm"), path + ".window_refine_margin_ppm"),
            get_nonempty_string(wire.at("prediction_config_sha256"),
                                path + ".prediction_config_sha256")));
    result->validate();
    return result;
}

void PipelineKernelContract::validate() const {
    if (projection_abi_version_ != kProjectionAbiVersion ||
        descriptor_version_ != kDescriptorVersion ||
        pipeline_protocol_version_ != kPipelineProtocolVersion ||
        model_version_ != kModelVersion ||
        timer_whitelist_version_ != kTimerWhitelistVersion) {
        fail("kernel_contract", "unsupported ABI/model/protocol/whitelist version");
    }
    if (window_tile_limit_ != 1 && window_tile_limit_ != 2 &&
        window_tile_limit_ != 4 && window_tile_limit_ != 8) {
        fail("kernel_contract.window_tile_limit", "must be one of 1, 2, 4, 8");
    }
    if (window_local_error_ppm_ > 10000) {
        fail("kernel_contract.window_local_error_ppm", "must be in [0, 10000]");
    }
    if (window_refine_margin_ppm_ > 1000000) {
        fail("kernel_contract.window_refine_margin_ppm", "must be in [0, 1000000]");
    }
    if (prediction_config_sha256_.size() != 64 ||
        !std::all_of(prediction_config_sha256_.begin(),
                     prediction_config_sha256_.end(),
                     [](unsigned char c) {
                         return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
                     })) {
        fail("kernel_contract.prediction_config_sha256",
             "must be 64 lowercase hexadecimal characters");
    }
}

bool PipelineKernelContract::operator==(
    const PipelineKernelContract& other) const {
    return projection_abi_version_ == other.projection_abi_version_ &&
           descriptor_version_ == other.descriptor_version_ &&
           pipeline_protocol_version_ == other.pipeline_protocol_version_ &&
           model_version_ == other.model_version_ &&
           timer_whitelist_version_ == other.timer_whitelist_version_ &&
           window_tile_limit_ == other.window_tile_limit_ &&
           window_local_error_ppm_ == other.window_local_error_ppm_ &&
           window_refine_margin_ppm_ == other.window_refine_margin_ppm_ &&
           prediction_config_sha256_ == other.prediction_config_sha256_;
}

PipelineWindowPlan PipelineKernel::plan_window(
    const PipelineKernelContract& contract,
    const uint32_t effective_limit,
    const uint32_t preferred_limit,
    const std::vector<uint64_t>& original_service_ticks,
    const std::vector<uint64_t>& hardware_minimum_ticks,
    const uint64_t local_span_ticks) {
    if (!valid_window_limit(effective_limit) ||
        effective_limit > contract.window_tile_limit()) {
        fail("effective window limit is unsupported by the run contract");
    }
    if (!valid_window_limit(preferred_limit)) {
        fail("preferred window limit must be one of 1, 2, 4, 8");
    }
    if (!hardware_minimum_ticks.empty() &&
        hardware_minimum_ticks.size() != original_service_ticks.size()) {
        fail("hardware minimum vector must match service vector length");
    }

    PipelineWindowPlan result;
    result.requested_limit = effective_limit;
    result.local_span_ticks = local_span_ticks;
    if (original_service_ticks.empty()) {
        result.unsupported_reason = "no_known_repeated_slices";
        return result;
    }

    const uint32_t preferred = std::min(preferred_limit, effective_limit);
    const uint32_t available = static_cast<uint32_t>(std::min<size_t>(
        original_service_ticks.size(), std::numeric_limits<uint32_t>::max()));
    uint32_t candidate = greatest_window_level(std::min(preferred, available));
    for (;;) {
        result.original_service_ticks.assign(original_service_ticks.begin(),
            original_service_ticks.begin() + candidate);
        unsigned __int128 total_wide = 0;
        for (const auto service : result.original_service_ticks) {
            total_wide += service;
        }
        if (total_wide > std::numeric_limits<uint64_t>::max()) {
            fail("window service total overflows uint64");
        }
        const uint64_t total = static_cast<uint64_t>(total_wide);
        const uint64_t quotient = total / candidate;
        const uint64_t remainder = total % candidate;
        result.assigned_service_ticks.clear();
        result.assigned_service_ticks.reserve(candidate);
        bool floors_satisfied = true;
        for (uint32_t index = 0; index < candidate; ++index) {
            const uint64_t assigned = quotient + (index < remainder ? 1 : 0);
            result.assigned_service_ticks.push_back(assigned);
            const uint64_t hardware_floor = hardware_minimum_ticks.empty()
                ? 0 : hardware_minimum_ticks[index];
            const uint64_t inherent_floor =
                result.original_service_ticks[index] == 0 ? 0 : 1;
            const uint64_t floor = std::max(hardware_floor, inherent_floor);
            if (result.original_service_ticks[index] < floor || assigned < floor) {
                floors_satisfied = false;
            }
        }

        __int128 prefix_delta = 0;
        unsigned __int128 max_prefix = 0;
        for (uint32_t index = 0; index < candidate; ++index) {
            prefix_delta += static_cast<__int128>(
                result.assigned_service_ticks[index]) -
                static_cast<__int128>(result.original_service_ticks[index]);
            const unsigned __int128 abs_delta = prefix_delta < 0
                ? static_cast<unsigned __int128>(-prefix_delta)
                : static_cast<unsigned __int128>(prefix_delta);
            max_prefix = std::max(max_prefix, abs_delta);
        }
        if (max_prefix > std::numeric_limits<uint64_t>::max()) {
            fail("local prefix error overflows uint64");
        }
        const uint64_t prefix_bound = static_cast<uint64_t>(max_prefix);
        const unsigned __int128 scaled_bound = max_prefix * 1000000U;
        const unsigned __int128 scaled_budget =
            static_cast<unsigned __int128>(contract.window_local_error_ppm()) *
            local_span_ticks;
        const bool within_prefix_budget = prefix_bound <=
                PipelineKernelContract::absolute_prefix_budget_ticks() ||
            scaled_bound <= scaled_budget;

        result.selected_count = candidate;
        result.original_total_ticks = total;
        result.max_prefix_delta_ticks = prefix_bound;
        result.exact = result.original_service_ticks ==
                       result.assigned_service_ticks;
        if (floors_satisfied && (candidate == 1 || within_prefix_budget)) {
            result.accepted = true;
            result.unsupported_reason.clear();
            return result;
        }
        if (candidate == 1) {
            result.unsupported_reason = floors_satisfied
                ? "exact_slice_below_service_floor"
                : "hardware_minimum_exceeds_exact_service";
            return result;
        }
        result.refinement_count = add_counter(
            result.refinement_count, 1, "refinement");
        result.unsupported_reason = !floors_satisfied
            ? "hardware_floor"
            : "local_prefix_budget";
        candidate /= 2;
    }
}

uint64_t PipelineKernel::original_service_at_external_tick(
    const PipelineWindowPlan& plan,
    const uint64_t external_elapsed_ticks) {
    if (!plan.accepted || plan.selected_count == 0 ||
        plan.original_service_ticks.size() != plan.selected_count ||
        plan.assigned_service_ticks.size() != plan.selected_count) {
        fail("service mapping requires a complete accepted plan");
    }
    unsigned __int128 external_total_wide = 0;
    unsigned __int128 original_total_wide = 0;
    for (std::size_t i = 0; i < plan.selected_count; ++i) {
        const uint64_t original = plan.original_service_ticks[i];
        const uint64_t assigned = plan.assigned_service_ticks[i];
        if (original != 0 && assigned == 0) {
            fail("positive original slice received zero external service");
        }
        external_total_wide += assigned;
        original_total_wide += original;
    }
    if (external_total_wide > UINT64_MAX ||
        original_total_wide > UINT64_MAX ||
        external_total_wide != plan.original_total_ticks) {
        fail("service mapping plan totals are inconsistent");
    }
    const uint64_t external_total =
        static_cast<uint64_t>(external_total_wide);
    if (external_elapsed_ticks > external_total) {
        fail("external service elapsed exceeds the committed plan");
    }
    if (external_total == 0) return 0;

    uint64_t external_prefix = 0;
    unsigned __int128 original_prefix = 0;
    for (std::size_t i = 0; i < plan.selected_count; ++i) {
        const uint64_t assigned = plan.assigned_service_ticks[i];
        const uint64_t original = plan.original_service_ticks[i];
        if (external_elapsed_ticks >= external_prefix + assigned) {
            external_prefix += assigned;
            original_prefix += original;
            continue;
        }
        const uint64_t local_external = external_elapsed_ticks - external_prefix;
        const unsigned __int128 local_original =
            static_cast<unsigned __int128>(original) * local_external /
            assigned;
        original_prefix += local_original;
        if (original_prefix > UINT64_MAX) {
            fail("mapped original service prefix overflows uint64");
        }
        return static_cast<uint64_t>(original_prefix);
    }
    if (original_prefix != original_total_wide) {
        fail("service mapping did not reach the original total");
    }
    return static_cast<uint64_t>(original_prefix);
}

PipelineWindowController::PipelineWindowController(
    const uint32_t configured_limit)
    : configured_limit_(configured_limit),
      current_limit_(greatest_window_level(std::min(4U, configured_limit))) {
    if (!valid_window_limit(configured_limit)) {
        fail("configured window limit must be one of 1, 2, 4, 8");
    }
}

PipelineWindowPlan PipelineWindowController::plan(
    const PipelineKernelContract& contract,
    const uint32_t effective_limit,
    const std::vector<uint64_t>& original_service_ticks,
    const std::vector<uint64_t>& hardware_minimum_ticks,
    const uint64_t local_span_ticks) const {
    if (configured_limit_ > contract.window_tile_limit()) {
        fail("controller limit exceeds immutable run contract");
    }
    if (!valid_window_limit(effective_limit) ||
        effective_limit > configured_limit_) {
        fail("effective limit exceeds controller configuration");
    }
    auto result = PipelineKernel::plan_window(
        contract, effective_limit, current_limit_, original_service_ticks,
        hardware_minimum_ticks, local_span_ticks);
    result.window_generation = generation_;
    result.tier_limit = current_limit_;
    return result;
}

void PipelineWindowController::record(const PipelineWindowPlan& plan,
                                      const bool committed) {
    if (!committed) return;
    if (plan.window_generation != generation_ ||
        plan.tier_limit != current_limit_) {
        fail("cannot commit a stale window generation or tier");
    }
    if (!plan.accepted || plan.selected_count == 0) {
        fail("cannot record a rejected window plan");
    }
    refinement_count_ = add_counter(refinement_count_,
                                    plan.refinement_count, "refinement");
    local_prefix_bound_max_ = std::max(
        local_prefix_bound_max_, plan.max_prefix_delta_ticks);
    if (!plan.exact) {
        window_count_ = add_counter(window_count_, 1, "window");
    }

    if (plan.selected_count < current_limit_) {
        current_limit_ = plan.selected_count;
        stable_windows_ = 0;
        return;
    }
    if (plan.exact) {
        // Exact fallback is valid service but is not a stable approximate
        // window. Do not promote the tier based on slices that did not use
        // the approximation path.
        stable_windows_ = 0;
        return;
    }
    stable_windows_ = add_counter(stable_windows_, 1, "stable-window");
    if (stable_windows_ == PipelineKernelContract::stable_window_count()) {
        if (current_limit_ < configured_limit_) {
            current_limit_ = std::min(configured_limit_, current_limit_ * 2);
        }
        stable_windows_ = 0;
  }
}

void PipelineWindowController::record_partial(
    const PipelineWindowPlan& plan) {
    if (plan.window_generation != generation_ ||
        plan.tier_limit != current_limit_) {
        fail("cannot commit a stale partial window generation or tier");
    }
    if (!plan.accepted || plan.selected_count == 0) {
        fail("cannot record a rejected partial window plan");
    }
    refinement_count_ = add_counter(refinement_count_,
                                    plan.refinement_count, "refinement");
    local_prefix_bound_max_ = std::max(
        local_prefix_bound_max_, plan.max_prefix_delta_ticks);
    if (!plan.exact) {
        window_count_ = add_counter(window_count_, 1, "window");
    }
    stable_windows_ = 0;
}

void PipelineWindowController::invalidate() {
    generation_ = add_counter(generation_, 1, "generation");
    stable_windows_ = 0;
}

nlohmann::json PipelineWindowPlan::to_json(
    const PipelineKernelContract& contract,
    const uint32_t effective_limit) const {
    nlohmann::json original = nlohmann::json::array();
    nlohmann::json assigned = nlohmann::json::array();
    for (const auto value : original_service_ticks) original.push_back(value);
    for (const auto value : assigned_service_ticks) assigned.push_back(value);
    return nlohmann::json{
        {"window_contract", {
            {"model_version", contract.model_version()},
            {"timer_whitelist_version", contract.timer_whitelist_version()},
            {"window_tile_limit", contract.window_tile_limit()},
            {"effective_window_tile_limit", effective_limit},
            {"window_local_error_ppm", contract.window_local_error_ppm()},
            {"window_refine_margin_ppm", contract.window_refine_margin_ppm()},
            {"stable_window_count", PipelineKernelContract::stable_window_count()},
            {"absolute_prefix_budget_ticks",
             PipelineKernelContract::absolute_prefix_budget_ticks()},
            {"average_remainder_rule",
             PipelineKernelContract::average_remainder_rule()}}},
        {"requested_limit", requested_limit},
        {"tier_limit", tier_limit},
        {"selected_count", selected_count},
        {"refinement_count", refinement_count},
        {"window_generation", window_generation},
        {"original_total_ticks", original_total_ticks},
        {"local_span_ticks", local_span_ticks},
        {"max_prefix_delta_ticks", max_prefix_delta_ticks},
        {"accepted", accepted}, {"exact", exact},
        {"unsupported_reason", unsupported_reason.empty()
             ? nlohmann::json(nullptr) : nlohmann::json(unsupported_reason)},
        {"original_service_ticks", std::move(original)},
        {"assigned_service_ticks", std::move(assigned)}};
}

nlohmann::json PipelineWindowController::counters_json(
    const PipelineKernelContract& contract,
    const uint32_t effective_limit) const {
    return nlohmann::json{
        {"model_version", contract.model_version()},
        {"timer_whitelist_version", contract.timer_whitelist_version()},
        {"window_tile_limit", contract.window_tile_limit()},
        {"effective_window_tile_limit", effective_limit},
        {"window_local_error_ppm", contract.window_local_error_ppm()},
        {"window_refine_margin_ppm", contract.window_refine_margin_ppm()},
        {"stable_window_count", PipelineKernelContract::stable_window_count()},
        {"absolute_prefix_budget_ticks",
         PipelineKernelContract::absolute_prefix_budget_ticks()},
        {"average_remainder_rule",
         PipelineKernelContract::average_remainder_rule()},
        {"window_count", window_count_},
        {"refinement_count", refinement_count_},
        {"local_prefix_bound_max", local_prefix_bound_max_},
        {"current_window_limit", current_limit_},
        {"stable_windows", stable_windows_},
        {"window_generation", generation_}};
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
