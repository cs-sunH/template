// joint_ledger_capi_test.cc -- C ABI surface test over the shared library:
// structural scan, a minimal full+delta delivery pair through the complete
// native receive path, query exports, and fail-closed error contract.
#include <cstdio>
#include <cstring>
#include <string>

#include "JointLedgerCApi.h"

static int failures = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); \
        ++failures; \
    } \
} while (0)

// Minimal v4 envelope with every required top-level key; arrays empty.
static std::string make_envelope(const std::string& kind, uint64_t delivery,
                                 uint64_t tick, uint64_t count,
                                 uint64_t seq_start, uint64_t seq_end,
                                 uint64_t base_version, uint64_t result_version,
                                 const char* producer = "cpp_online") {
    char buffer[1024];
    std::snprintf(buffer, sizeof(buffer),
        "{\"producer\":\"%s\",\"schema_version\":4,\"protocol_version\":4,"
        "\"bridge_protocol\":\"bridge-v4-20261011\","
        "\"work_ledger_schema\":\"workledger-v4-20261011\","
        "\"epoch\":\"e1\",\"delivery_sequence\":%llu,"
        "\"base_watermark\":%llu,\"logical_seq_start\":%llu,"
        "\"logical_seq_end\":%llu,\"logical_event_count\":%llu,"
        "\"replay_token\":\"%s/e1/%llu\",\"descriptor_version\":1,"
        "\"base_state_version\":%llu,\"result_state_version\":%llu,"
        "\"snapshot_kind\":\"%s\",\"tick\":%llu,"
        "\"descriptor_commits\":[],\"node_transitions\":[],"
        "\"pipeline_transition_runs\":[],\"cursor_patches\":[],"
        "\"active_resources\":[]}",
        producer, (unsigned long long)delivery,
        (unsigned long long)(seq_start == 0 ? 1 : seq_start - 1),
        (unsigned long long)seq_start, (unsigned long long)seq_end,
        (unsigned long long)count, producer,
        (unsigned long long)delivery,
        (unsigned long long)base_version,
        (unsigned long long)result_version, kind.c_str(),
        (unsigned long long)tick);
    return buffer;
}

int main() {
    // ---- structural scan -------------------------------------------------
    const std::string request =
        "{\"schema_version\":2,\"tick\":7,\"work_ledger\":{\"a\":[1,2]},"
        "\"reasons\":[]}";
    JLScanResult scan;
    CHECK(jl_scan_request(request.data(), request.size(), &scan) == JL_OK);
    const std::string outer = request.substr(0, scan.member_start)
        + request.substr(scan.member_end);
    CHECK(outer == "{\"schema_version\":2,\"tick\":7,\"reasons\":[]}");
    const std::string segment = request.substr(
        scan.value_start, scan.value_end - scan.value_start);
    CHECK(segment == "{\"a\":[1,2]}");

    // ---- full + delta deliveries through the native receiver -------------
    JLedger* ledger = jl_create();
    CHECK(ledger != nullptr);

    const std::string full = make_envelope("full", 0, 5, 0, 1, 0, 0, 0);
    int32_t prepared = -1;
    uint64_t event_count = 99, seq_end = 99, result_version = 99;
    int code = jl_prepare(ledger, full.data(), full.size(), 0, 5, &prepared,
                          &event_count, &seq_end, &result_version);
    if (code != JL_OK) {
        char message[256];
        jl_last_error(ledger, message, sizeof(message));
        std::fprintf(stderr, "prepare(full) failed: %s\n", message);
    }
    CHECK(code == JL_OK);
    CHECK(prepared >= 0);
    CHECK(event_count == 0);
    CHECK(jl_commit(ledger, prepared) == JL_OK);

    JLCounters counters;
    CHECK(jl_counters(ledger, &counters) == JL_OK);
    CHECK(counters.full_load_count == 1);
    CHECK(counters.delta_load_count == 0);
    CHECK(counters.pipeline_delivery_sequence == 0);
    CHECK(counters.last_snapshot_full == 1);

    const std::string delta = make_envelope("delta", 1, 6, 0, 1, 0, 0, 0);
    code = jl_prepare(ledger, delta.data(), delta.size(), 1, 6, &prepared,
                      &event_count, &seq_end, &result_version);
    if (code != JL_OK) {
        char message[256];
        jl_last_error(ledger, message, sizeof(message));
        std::fprintf(stderr, "prepare(delta) failed: %s\n", message);
    }
    CHECK(code == JL_OK);
    CHECK(jl_commit(ledger, prepared) == JL_OK);
    CHECK(jl_counters(ledger, &counters) == JL_OK);
    CHECK(counters.delta_load_count == 1);

    // Query exports parse as JSON objects.
    size_t capacity = 0;
    CHECK(jl_pricing_json(ledger, nullptr, &capacity) == JL_ERR_RANGE);
    std::string pricing(capacity, '\0');
    CHECK(jl_pricing_json(ledger, pricing.data(), &capacity) == JL_OK);
    CHECK(pricing.find("\"work\"") != std::string::npos);
    CHECK(pricing.find("\"pipeline\"") != std::string::npos);

    capacity = 0;
    CHECK(jl_conservation_json(ledger, nullptr, &capacity) == JL_ERR_RANGE);
    std::string conservation(capacity, '\0');
    CHECK(jl_conservation_json(ledger, conservation.data(), &capacity)
          == JL_OK);
    CHECK(conservation.find("\"conserved\"") != std::string::npos);

    // ---- fail-closed contract -------------------------------------------
    // Bad producer identity is rejected before any state change.
    const std::string bad = make_envelope("delta", 2, 7, 0, 1, 0, 0, 0,
                                          "someone_else");
    code = jl_prepare(ledger, bad.data(), bad.size(), 2, 7, &prepared,
                      &event_count, &seq_end, &result_version);
    CHECK(code == JL_ERR_PROTOCOL);
    // Delivery sequence mismatch with the outer frame is rejected.
    const std::string seq_mismatch = make_envelope("delta", 3, 7, 0, 1, 0,
                                                   0, 0);
    CHECK(jl_prepare(ledger, seq_mismatch.data(), seq_mismatch.size(), 2, 7,
                     &prepared, &event_count, &seq_end, &result_version)
          == JL_ERR_PROTOCOL);
    // State remains usable: the next legal delivery still applies.
    const std::string next = make_envelope("delta", 2, 7, 0, 1, 0, 0, 0);
    code = jl_prepare(ledger, next.data(), next.size(), 2, 7, &prepared,
                      &event_count, &seq_end, &result_version);
    if (code != JL_OK) {
        char message[256];
        jl_last_error(ledger, message, sizeof(message));
        std::fprintf(stderr, "prepare(next legal) failed: %s\n", message);
    }
    CHECK(code == JL_OK);
    CHECK(jl_commit(ledger, prepared) == JL_OK);
    CHECK(jl_counters(ledger, &counters) == JL_OK);
    CHECK(counters.delta_load_count == 2);

    // Garbage bytes are rejected without crashing.
    const char* garbage = "{\"producer\":";
    CHECK(jl_prepare(ledger, garbage, std::strlen(garbage), 3, 8, &prepared,
                     &event_count, &seq_end, &result_version)
          == JL_ERR_PROTOCOL);

    // API misuse on unknown slots.
    CHECK(jl_commit(ledger, 12345) == JL_ERR_PROTOCOL);

    jl_destroy(ledger);
    jl_destroy(nullptr);

    if (failures == 0) {
        std::printf("joint_ledger_capi_test PASS\n");
        return 0;
    }
    std::printf("joint_ledger_capi_test FAIL (%d)\n", failures);
    return 1;
}
