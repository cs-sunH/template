/******************************************************************************
Exception-safe C ABI for joint native projection.
******************************************************************************/

#include "astra-sim/workload/execution_driven/NativeProjectionCApi.hh"

#include "astra-sim/workload/execution_driven/NativeProjection.hh"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <string>

namespace {

struct ContextHandle {
    std::shared_ptr<const AstraSim::ExecutionDriven::NativeProjectionContext> context;
    std::unique_ptr<AstraSim::ExecutionDriven::NativeProjectionWorldSession> world;
    std::string error;
};

thread_local std::string create_error;

bool valid_buffer(const char* data, size_t len) {
    return data != nullptr || len == 0;
}

int write_error(ContextHandle* context, const char* message, int code) {
    if (context != nullptr) context->error = message ? message : "native projection error";
    return code;
}

}  // namespace

extern "C" {

uint32_t jproj_abi_version(void) { return 1; }

int jproj_create(const char* snapshot, size_t len, void** out_context) {
    create_error.clear();
    if (out_context == nullptr) {
        create_error = "jproj_create: out_context is null";
        return -1;
    }
    *out_context = nullptr;
    if (!valid_buffer(snapshot, len) || snapshot == nullptr) {
        create_error = "jproj_create: snapshot pointer/length is invalid";
        return -1;
    }
    try {
        auto envelope = nlohmann::json::parse(snapshot, snapshot + len);
        auto handle = std::make_unique<ContextHandle>();
        handle->context =
            AstraSim::ExecutionDriven::NativeProjectionContext::parse(envelope);
        // The frozen snapshot may carry barrier signallers owned by a
        // candidate that is supplied only to jproj_project. Constructing a
        // mutable world here would validate those references before the
        // candidate has been merged. Keep context creation immutable/lazy;
        // project() validates the complete merged world at the query point.
        *out_context = handle.release();
        return 0;
    } catch (const std::exception& ex) {
        create_error = ex.what();
        return -1;
    } catch (...) {
        create_error = "jproj_create: unknown native exception";
        return -2;
    }
}

int jproj_project(void* opaque, const char* candidate, size_t candidate_len,
                  const char* options, size_t options_len,
                  char** out_result, size_t* out_len) {
    auto* handle = static_cast<ContextHandle*>(opaque);
    if (handle == nullptr) return -1;
    handle->error.clear();
    if (out_result == nullptr || out_len == nullptr) {
        return write_error(handle, "jproj_project: result out parameters are null", -1);
    }
    *out_result = nullptr;
    *out_len = 0;
    if (!valid_buffer(candidate, candidate_len) || candidate == nullptr ||
        !valid_buffer(options, options_len) || options == nullptr) {
        return write_error(handle, "jproj_project: input pointer/length is invalid", -1);
    }
    try {
        auto candidate_json = nlohmann::json::parse(candidate, candidate + candidate_len);
        auto options_json = nlohmann::json::parse(options, options + options_len);
        if (!candidate_json.is_object() || !candidate_json.contains("candidates")
                || !candidate_json.at("candidates").is_array()) {
            return write_error(handle,
                "candidate payload must be an object with a 'candidates' "
                "array of WorkItems", -1);
        }
        auto parsed_options = AstraSim::ExecutionDriven::NativeProjectionOptions::parse(
            options_json);
        auto result = AstraSim::ExecutionDriven::NativeProjection::project(
            *handle->context, candidate_json, parsed_options);
        const std::string serialized = result.dump();
        char* buffer = static_cast<char*>(std::malloc(serialized.size() + 1));
        if (buffer == nullptr) {
            return write_error(handle, "jproj_project: result allocation failed", -2);
        }
        std::memcpy(buffer, serialized.data(), serialized.size());
        buffer[serialized.size()] = '\0';
        *out_result = buffer;
        *out_len = serialized.size();
        const auto status = result.value("status", std::string());
        if (status == "degraded" || status == "model_unavailable") return 1;
        return 0;
    } catch (const std::exception& ex) {
        return write_error(handle, ex.what(), -1);
    } catch (...) {
        return write_error(handle, "jproj_project: unknown native exception", -2);
    }
}

int jproj_world_operation(void* opaque, const char* request, size_t request_len,
                          char** out_result, size_t* out_len) {
    auto* handle = static_cast<ContextHandle*>(opaque);
    if (handle == nullptr) return -1;
    handle->error.clear();
    if (out_result == nullptr || out_len == nullptr) {
        return write_error(handle, "jproj_world_operation: result out parameters are null", -1);
    }
    *out_result = nullptr;
    *out_len = 0;
    if (!valid_buffer(request, request_len) || request == nullptr) {
        return write_error(handle, "jproj_world_operation: input pointer/length is invalid", -1);
    }
    try {
        if (!handle->world) {
            handle->world = std::make_unique<
                AstraSim::ExecutionDriven::NativeProjectionWorldSession>(handle->context);
        }
        auto request_json = nlohmann::json::parse(request, request + request_len);
        auto result = handle->world->operation(request_json);
        const std::string serialized = result.dump();
        char* buffer = static_cast<char*>(std::malloc(serialized.size() + 1));
        if (buffer == nullptr) {
            return write_error(handle, "jproj_world_operation: result allocation failed", -2);
        }
        std::memcpy(buffer, serialized.data(), serialized.size());
        buffer[serialized.size()] = '\0';
        *out_result = buffer;
        *out_len = serialized.size();
        return 0;
    } catch (const std::exception& ex) {
        return write_error(handle, ex.what(), -1);
    } catch (...) {
        return write_error(handle, "jproj_world_operation: unknown native exception", -2);
    }
}

void jproj_free_result(char* result) { std::free(result); }

void jproj_destroy(void* opaque) {
    delete static_cast<ContextHandle*>(opaque);
}

const char* jproj_last_error(void* opaque) {
    if (opaque == nullptr) return create_error.c_str();
    return static_cast<ContextHandle*>(opaque)->error.c_str();
}

}  // extern "C"
