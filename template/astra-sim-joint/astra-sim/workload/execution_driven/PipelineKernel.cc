/******************************************************************************
Run-scoped kernel contract implementation for the die compute backend
(the bounded window algebra and its precision controller were retired with
the K>1 timer-window approximation on 2026-10-16; this file now holds only
the PipelineKernelContract parse/validate/equality implementation that
every GraphBatch freezes at commit time).
******************************************************************************/

#include "astra-sim/workload/execution_driven/PipelineKernelContract.hh"

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
    std::string timer_whitelist_version,
    std::string prediction_config_sha256)
    : projection_abi_version_(projection_abi_version),
      descriptor_version_(descriptor_version),
      pipeline_protocol_version_(pipeline_protocol_version),
      model_version_(std::move(model_version)),
      timer_whitelist_version_(std::move(timer_whitelist_version)),
      prediction_config_sha256_(std::move(prediction_config_sha256)) {}

std::shared_ptr<const PipelineKernelContract> PipelineKernelContract::parse(
    const nlohmann::json& wire) {
    const std::string path = "kernel_contract";
    require_exact_keys(wire, path,
                       {"projection_abi_version", "descriptor_version",
                        "pipeline_protocol_version", "model_version",
                        "timer_whitelist_version",
                        "prediction_config_sha256"});
    auto result = std::shared_ptr<const PipelineKernelContract>(
        new PipelineKernelContract(
            get_u32(wire.at("projection_abi_version"), path + ".projection_abi_version"),
            get_u32(wire.at("descriptor_version"), path + ".descriptor_version"),
            get_u32(wire.at("pipeline_protocol_version"), path + ".pipeline_protocol_version"),
            get_nonempty_string(wire.at("model_version"), path + ".model_version"),
            get_nonempty_string(wire.at("timer_whitelist_version"),
                                path + ".timer_whitelist_version"),
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
           prediction_config_sha256_ == other.prediction_config_sha256_;
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
