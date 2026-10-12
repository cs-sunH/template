// Capi.cc -- C ABI boundary; no C++ exception ever crosses it.
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "joint_ledger/DualLedger.hh"
#include "joint_ledger/JsonScanner.hh"
#include "JointLedgerCApi.h"
#include "joint_ledger/LedgerTypes.hh"

namespace {

// Scan errors carry no handle; the last message is kept in a process-local
// slot guarded by a mutex (jl_scan_request is called once per delivery).
std::mutex g_error_mutex;
std::string g_last_error;

void set_error(const std::string& message) {
    std::lock_guard<std::mutex> lock(g_error_mutex);
    g_last_error = message;
}

template <typename Body>
int guarded(const JLedger* ledger, Body body) {
    try {
        return body();
    } catch (const jointledger::LedgerError& error) {
        if (ledger != nullptr) {
            try {
                const_cast<JLedger*>(ledger)->message = error.what();
            } catch (...) {
                set_error(error.what());
            }
        } else {
            set_error(error.what());
        }
        return JL_ERR_PROTOCOL;
    } catch (const std::bad_alloc&) {
        set_error("allocation failure");
        return JL_ERR_MEMORY;
    } catch (const std::exception& error) {
        set_error(std::string("internal error: ") + error.what());
        return JL_ERR_PROTOCOL;
    } catch (...) {
        set_error("unknown internal error");
        return JL_ERR_PROTOCOL;
    }
}

int write_json(const std::string& payload, char* buffer, size_t* capacity) {
    if (capacity == nullptr) {
        return JL_ERR_ARG;
    }
    const size_t needed = payload.size() + 1;
    if (buffer == nullptr || *capacity < needed) {
        // Size query (null buffer) or undersized buffer: report the need.
        *capacity = needed;
        return JL_ERR_RANGE;
    }
    std::memcpy(buffer, payload.data(), payload.size());
    buffer[payload.size()] = '\0';
    *capacity = needed;
    return JL_OK;
}

}  // namespace

struct JLedger {
    jointledger::DualLedger core;
    std::string message;
};

extern "C" {

JLedger* jl_create(void) {
    JLedger* ledger = new (std::nothrow) JLedger();
    return ledger;
}

void jl_destroy(JLedger* ledger) { delete ledger; }

int jl_scan_request(const char* data, size_t length, JLScanResult* out) {
    return guarded(nullptr, [&]() -> int {
        if (data == nullptr || out == nullptr) return JL_ERR_ARG;
        const jointledger::ScanResult result =
            jointledger::scan_request(data, length);
        out->member_start = result.member_start;
        out->member_end = result.member_end;
        out->value_start = result.value_start;
        out->value_end = result.value_end;
        return JL_OK;
    });
}

int jl_prepare(JLedger* ledger, const char* segment, size_t length,
               uint64_t expected_delivery, uint64_t expected_tick,
               int32_t* prepared, uint64_t* event_count, uint64_t* seq_end,
               uint64_t* result_state_version) {
    if (ledger == nullptr || segment == nullptr || prepared == nullptr) {
        return JL_ERR_ARG;
    }
    return guarded(ledger, [&]() -> int {
        *prepared = ledger->core.prepare(
            segment, length, expected_delivery, expected_tick, event_count,
            seq_end, result_state_version);
        return JL_OK;
    });
}

int jl_commit(JLedger* ledger, int32_t prepared) {
    if (ledger == nullptr) return JL_ERR_ARG;
    return guarded(ledger, [&]() -> int {
        ledger->core.commit(prepared);
        return JL_OK;
    });
}

int jl_discard_prepared(JLedger* ledger, int32_t prepared) {
    if (ledger == nullptr) return JL_ERR_ARG;
    return guarded(ledger, [&]() -> int {
        ledger->core.discard(prepared);
        return JL_OK;
    });
}

int jl_last_error(const JLedger* ledger, char* buffer, size_t buffer_size) {
    if (buffer == nullptr || buffer_size == 0) return JL_ERR_ARG;
    std::lock_guard<std::mutex> lock(g_error_mutex);
    // Prefer the handle-local message, falling back to the global slot for
    // handle-less scan failures.
    std::string message = g_last_error;
    if (ledger != nullptr && !ledger->message.empty()) {
        message = ledger->message;
    }
    const size_t copy = message.size() < buffer_size - 1
        ? message.size() : buffer_size - 1;
    std::memcpy(buffer, message.data(), copy);
    buffer[copy] = '\0';
    return static_cast<int>(message.size());
}

const char* jl_last_error_direct(const JLedger* ledger) {
    if (ledger != nullptr && !ledger->message.empty()) return ledger->message.c_str();
    std::lock_guard<std::mutex> lock(g_error_mutex);
    return g_last_error.c_str();
}

int jl_counters(const JLedger* ledger, JLCounters* out) {
    if (ledger == nullptr || out == nullptr) return JL_ERR_ARG;
    return guarded(ledger, [&]() -> int {
        const auto& work = ledger->core.work();
        const auto& pipeline = ledger->core.pipeline();
        out->version = work.version();
        out->event_watermark = work.event_watermark();
        out->last_tick = work.last_tick();
        out->last_delivery_sequence = work.last_delivery_sequence();
        out->full_load_count = work.full_load_count();
        out->delta_load_count = work.delta_load_count();
        out->applied_event_count = work.applied_event_count();
        out->replayed_event_count = work.replayed_event_count();
        out->pipeline_delivery_sequence =
            static_cast<int64_t>(pipeline.delivery_sequence());
        out->pipeline_watermark = pipeline.watermark();
        out->pipeline_state_version = pipeline.state_version();
        out->active_works = ledger->core.active_work_count();
        const auto& phase = ledger->core.last_phase_ns();
        out->parse_ns = phase.parse;
        out->stage_ns = phase.stage;
        out->commit_ns = phase.commit;
        out->last_snapshot_full =
            ledger->core.last_snapshot_full() ? 1 : 0;
        return JL_OK;
    });
}

int jl_pricing_json(const JLedger* ledger, char* buffer, size_t* capacity) {
    if (ledger == nullptr || capacity == nullptr) return JL_ERR_ARG;
    return guarded(ledger, [&]() -> int {
        jointledger::Json payload;
        payload["work"] = ledger->core.work().pricing_json();
        payload["pipeline"] = ledger->core.pipeline().pricing_json();
        return write_json(payload.dump(), buffer, capacity);
    });
}

int jl_conservation_json(const JLedger* ledger, char* buffer,
                         size_t* capacity) {
    if (ledger == nullptr || capacity == nullptr) return JL_ERR_ARG;
    return guarded(ledger, [&]() -> int {
        return write_json(ledger->core.work().conservation_json().dump(),
                          buffer, capacity);
    });
}

int jl_works_export_json(const JLedger* ledger, char* buffer,
                         size_t* capacity) {
    if (ledger == nullptr || capacity == nullptr) return JL_ERR_ARG;
    return guarded(ledger, [&]() -> int {
        return write_json(ledger->core.work().works_export_json().dump(),
                          buffer, capacity);
    });
}

int jl_pipeline_snapshot_json(const JLedger* ledger, char* buffer,
                              size_t* capacity) {
    if (ledger == nullptr || capacity == nullptr) return JL_ERR_ARG;
    return guarded(ledger, [&]() -> int {
        return write_json(ledger->core.pipeline().snapshot_state_json().dump(),
                          buffer, capacity);
    });
}

}  // extern "C"
