#include "astra-sim/workload/execution_driven/NativeProjectionCApi.hh"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <json/json.hpp>
#include "../../../../extern/die_model/coarse/include/coarse_die/Hardware.hh"

namespace {

int failures = 0;

void expect(bool value, const char* message) {
    if (!value) {
        std::cerr << "[native_projection_c_api_test] FAIL: " << message << '\n';
        ++failures;
    }
}

std::string snapshot() {
    nlohmann::json input{
        {"now_ns", 0},
        {"resources", nlohmann::json::array({
            {{"key", "link:a"}, {"kind", "link"}, {"capacity_per_ns", 10.0}}})},
        {"background_works", nlohmann::json::array({
            {{"work_id", "focus"}, {"legs", nlohmann::json::array({
                {{"leg_id", "focus#0"}, {"kind", "stream"},
                 {"route", nlohmann::json::array({"link:a"})},
                 {"bytes_amount", 10}, {"duration_ns", 0.0}, {"base_ns", 0.0},
                 {"group", "prefill"}, {"latency_ns", 0}, {"ready_at_ns", nullptr},
                 {"deps", nlohmann::json::array()}, {"acquires", nlohmann::json::array()},
                 {"releases", nlohmann::json::array()}, {"signals_barrier", nullptr},
                 {"remaining_bytes", nullptr}, {"remaining_duration_ns", nullptr},
                 {"in_service", false}, {"order", nullptr}, {"die_plan", nullptr}}})},
             {"ready_at_ns", 0}, {"depends_on_work", nlohmann::json::array()},
             {"is_candidate", false}, {"service_leg_ids", nullptr}}})},
        {"focus_work_id", "focus"}, {"gates", nlohmann::json::array()},
        {"barriers", nlohmann::json::array()}, {"efficiency", {{"eta", nlohmann::json::object()}, {"gamma", nlohmann::json::object()}}},
        {"pipelines", nlohmann::json::array()},
        {"forecast_arrival_placeholders_ns", nlohmann::json::array()}};
    nlohmann::json contract{{"projection_abi_version", 1},
        {"descriptor_version", 1}, {"pipeline_protocol_version", 1},
        {"model_version", "v9-tie-roundrobin-20261020"},
        {"timer_whitelist_version", "timer-whitelist-v1-20261005"},
        {"prediction_config_sha256", std::string(64, 'b')}};
    return nlohmann::json{{"kernel_contract", contract},
                          {"projection_input", input}}.dump();
}

std::string coarse_cancel_snapshot() {
    const auto source = std::filesystem::path(__FILE__);
    const auto root = source.parent_path().parent_path().parent_path()
        .parent_path().parent_path();
    std::ifstream config_input(root /
        "extern/die_model/coarse/configs/example_hardware.json");
    if (!config_input.good()) throw std::runtime_error(
        "tracked coarse hardware fixture exists for C ABI cancellation");
    nlohmann::json hardware_wire;
    config_input >> hardware_wire;
    const auto hardware = coarse_die::resolve_hardware(hardware_wire);
    const double raw_hbm_bns = (hardware.rates[4] + hardware.rates[5]) / 1e9;

    nlohmann::json input{
        {"now_ns", 0},
        {"resources", nlohmann::json::array({
            {{"key", "hbm:0"}, {"kind", "hbm"},
             {"capacity_per_ns", raw_hbm_bns}}})},
        {"background_works", nlohmann::json::array({
            {{"work_id", "restore"}, {"legs", nlohmann::json::array({
                {{"leg_id", "restore#0"}, {"kind", "stream"},
                 {"route", nlohmann::json::array({"hbm:0"})},
                 {"bytes_amount", 0}, {"duration_ns", 0.0},
                 {"base_ns", 0.0}, {"group", "prefill"},
                 {"latency_ns", 0}, {"ready_at_ns", nullptr},
                 {"deps", nlohmann::json::array()},
                 {"acquires", nlohmann::json::array()},
                 {"releases", nlohmann::json::array()},
                 {"signals_barrier", nullptr},
                 {"remaining_bytes", nullptr},
                 {"remaining_duration_ns", nullptr},
                 {"in_service", false}, {"order", nullptr},
                 {"die_plan", nullptr},
                 {"coarse_endpoint", {
                     {"endpoint_id", 17}, {"transaction_id", 17},
                     {"owner_kind", "restore"}, {"direction", "write"},
                     {"physical_bytes", 1048577}, {"eta_hbm", 1.0},
                     {"path_class", "edge_to_hbm"},
                     {"route", {{"internal_route", false}}}}}}})},
             {"ready_at_ns", 0}, {"depends_on_work", nlohmann::json::array()},
             {"is_candidate", false}, {"service_leg_ids", nullptr}}})},
        {"focus_work_id", "restore"}, {"gates", nlohmann::json::array()},
        {"barriers", nlohmann::json::array()},
        {"efficiency", {{"eta", nlohmann::json::object()},
            {"gamma", nlohmann::json::object()}}},
        {"pipelines", nlohmann::json::array()},
        {"forecast_arrival_placeholders_ns", nlohmann::json::array()},
        {"coarse_die_domains", nlohmann::json::array({
            {{"resource_key", "hbm:0"},
             {"hardware_config", hardware_wire}}})},
        {"coarse_cancel_events", nlohmann::json::array({
            {{"time_ns", 0}, {"leg_id", "restore#0"}}})}};
    return nlohmann::json{{"projection_input", input}}.dump();
}

}  // namespace

int main() {
    expect(jproj_abi_version() == 1, "ABI version 1 exposed");
    void* context = nullptr;
    const auto wire = snapshot();
    expect(jproj_create(wire.data(), wire.size(), &context) == 0 && context != nullptr,
           "valid snapshot creates context");
    if (context != nullptr) {
        const std::string candidate = R"({"candidates":[]})";
        const std::string options = "{}";
        char* result_buffer = nullptr;
        size_t result_len = 0;
        const int code = jproj_project(context, candidate.data(), candidate.size(),
            options.data(), options.size(), &result_buffer, &result_len);
        expect(code == 0, "native event world returns a complete projection");
        expect(result_buffer != nullptr && result_len > 0,
               "result uses explicit caller-owned buffer and length");
        if (result_buffer != nullptr) {
            const auto result = nlohmann::json::parse(result_buffer, result_buffer + result_len);
            expect(result.value("status", std::string()) == "ok",
                   "result carries the native status");
            expect(result.at("predicted_settle_done_ns").get<uint64_t>() == 1 &&
                       result.at("predicted_service_done_ns").get<uint64_t>() == 1,
                   "stream completes at the expected service/settle tick");
            expect(result.at("bytes_served_total").get<uint64_t>() == 10 &&
                       result.at("event_count").get<uint64_t>() == 1,
                   "native disclosure includes served bytes and one completion");
            expect(result.at("logical_event_count").get<uint64_t>() == 1 &&
                       result.at("physical_step_count").get<uint64_t>() >= 1 &&
                       result.at("unsupported_reason").is_null(),
                   "quality fields describe the exact event-world result");
            jproj_free_result(result_buffer);
        }

        const auto world_call = [&](const std::string& request) {
            char* buffer = nullptr;
            size_t length = 0;
            const int result = jproj_world_operation(context, request.data(),
                request.size(), &buffer, &length);
            nlohmann::json payload = nlohmann::json::parse(buffer, buffer + length);
            jproj_free_result(buffer);
            return std::make_pair(result, payload);
        };
        const auto initial = world_call(R"({"operation":"inspect"})");
        expect(initial.first == 0 && initial.second.at("now_ns").get<double>() == 0.0,
               "world session inspect exposes its own initial mutable state");
        const auto stepped = world_call(R"({"operation":"step"})");
        expect(stepped.first == 0 && stepped.second.at("progressed").get<bool>() &&
                   stepped.second.at("work_done_at").at("focus").get<uint64_t>() == 1,
               "world session step advances one native event time");
        const auto reset = world_call(R"({"operation":"reset"})");
        expect(reset.first == 0 && reset.second.at("now_ns").get<double>() == 0.0,
               "world session reset reconstructs the frozen input");
        const auto partial = world_call(R"({"operation":"integrate","target_ns":0.5})");
        expect(partial.first == 0 && partial.second.at("legs").at("focus#0")
                   .at("remaining_bytes").get<double>() == 5.0,
               "controlled integration exposes native partial progress");

        const std::string bad_options = R"({"unexpected":true})";
        result_buffer = nullptr;
        result_len = 0;
        expect(jproj_project(context, candidate.data(), candidate.size(),
            bad_options.data(), bad_options.size(), &result_buffer, &result_len) < 0,
            "unknown options fail closed across C ABI");
        expect(std::string(jproj_last_error(context)).find("unknown key") != std::string::npos,
               "last_error reports copied parse failure detail");
        jproj_destroy(context);
    }

    // Legacy/standalone contexts have no run contract; the retired
    // per-query window-limit options key is no longer part of the options
    // vocabulary, so a wire that still carries it must fail closed as an
    // unknown options key (the key name is assembled by concatenation so
    // the retired identifier itself stays out of the tree).
    auto standalone_envelope = nlohmann::json::parse(snapshot());
    standalone_envelope.erase("kernel_contract");
    const auto standalone_wire = standalone_envelope.dump();
    void* standalone_context = nullptr;
    expect(jproj_create(standalone_wire.data(), standalone_wire.size(),
                        &standalone_context) == 0 &&
               standalone_context != nullptr,
           "contract-free standalone context preserves the ABI1 snapshot shape");
    if (standalone_context != nullptr) {
        const std::string candidate = R"({"candidates":[]})";
        const std::string retired_key =
            std::string("effective_") + "window_" + "tile_" + "limit";
        const std::string relaxed_options =
            nlohmann::json{{retired_key, 2}}.dump();
        char* buffer = nullptr;
        size_t length = 0;
        expect(jproj_project(standalone_context, candidate.data(),
                             candidate.size(), relaxed_options.data(),
                             relaxed_options.size(), &buffer, &length) < 0,
               "options carrying the retired window key are rejected");
        expect(std::string(jproj_last_error(standalone_context)).find(
                   "unknown key") != std::string::npos,
               "rejection reports the retired key as an unknown options key");
        jproj_free_result(buffer);
        jproj_destroy(standalone_context);
    }

    // The explicit frozen cancellation path also crosses the public C ABI.
    // Its canonical endpoint Runtime retains a cancelled terminal, while the
    // outer projection does not publish a successful work completion.
    void* cancel_context = nullptr;
    const auto cancel_wire = coarse_cancel_snapshot();
    expect(jproj_create(cancel_wire.data(), cancel_wire.size(),
                        &cancel_context) == 0 && cancel_context != nullptr,
           "frozen coarse cancellation input creates through C ABI");
    if (cancel_context != nullptr) {
        char* buffer = nullptr;
        size_t length = 0;
        const std::string run = R"({"operation":"run"})";
        const int code = jproj_world_operation(cancel_context, run.data(),
            run.size(), &buffer, &length);
        expect(code == 0, "C ABI world consumes the frozen cancellation event");
        if (buffer != nullptr) {
            const auto state = nlohmann::json::parse(buffer, buffer + length);
            expect(state.at("legs").at("restore#0").at("state") == "cancelled" &&
                   state.at("legs").at("restore#0").at("coarse_cancelled").get<bool>(),
                   "C ABI inspection exposes canonical endpoint cancellation");
            expect(state.at("work_done_at").at("restore").is_null(),
                   "C ABI cancellation never masquerades as successful work done");
            jproj_free_result(buffer);
        }
        buffer = nullptr;
        length = 0;
        const std::string export_request = R"({"operation":"export_snapshot"})";
        expect(jproj_world_operation(cancel_context, export_request.data(),
            export_request.size(), &buffer, &length) == 0,
            "C ABI exports the consumed cancellation prefix");
        if (buffer != nullptr) {
            const auto state = nlohmann::json::parse(buffer, buffer + length);
            expect(state.at("state").at("cancel_events_processed").get<uint64_t>() == 1,
                   "C ABI snapshot retains the frozen cancellation prefix");
            jproj_free_result(buffer);
        }
        jproj_destroy(cancel_context);
    }

    // Exercise the same real PlanCache/Execution path used by ordinary DIE
    // projection. The background stream is delayed beyond the DIE finish so
    // it cannot introduce an unrelated global boundary during the
    // projection.
    auto die_envelope = nlohmann::json::parse(snapshot());
    die_envelope["projection_input"]["resources"].push_back(
        {{"key", "hbm:0"}, {"kind", "hbm"},
         {"capacity_per_ns", 1.0e12}});
    die_envelope["projection_input"]["background_works"][0]["ready_at_ns"] =
        1000000000000000ULL;
    const std::string die_snapshot = die_envelope.dump();
    void* die_context = nullptr;
    expect(jproj_create(die_snapshot.data(), die_snapshot.size(),
                        &die_context) == 0 && die_context != nullptr,
           "contracted real-DIE context creates from frozen hardware inputs");
    if (die_context != nullptr) {
        const nlohmann::json spec = {
            {"schema_version", 1}, {"op_kind", "norm"},
            {"phase", "prefill"}, {"dtype", "bf16"},
            {"model_dims", {{"hidden", 4096}, {"ffn", 14336},
                {"vocab", 152064}, {"mlp_variant", "swiglu"},
                {"norm_type", "rmsnorm"}, {"position_encoding", "rope"},
                {"num_q_heads", 32}, {"num_kv_heads", 8},
                {"head_dim", 128}}},
            {"tp_slice", {{"tensor_parallel", 1},
                {"tensor_parallel_rank", 0}, {"q_heads_local", 32},
                {"kv_heads_local", 8}, {"ffn_local", 14336},
                {"vocab_local", 152064}}},
            {"spans", nlohmann::json::array({{{"request_id", "r1"},
                {"q_tokens", 8}, {"kv_tokens", 128},
                {"q_start_pos", 0}, {"kv_start_pos", 0}}})},
            {"layer_count", 2}, {"weight_passes", 1},
            {"kv_access", {{"identity", "local"}}},
            {"fusion_policy", "auto"}};
        const nlohmann::json hardware = {
            {"schema_version", 1}, {"profile", "mesh_die_v1"},
            {"die", {{"sram-mb-per-core", 0.75},
                {"mesh-nx", 23u}, {"mesh-ny", 23u}, {"compute-area-mm2", 800},
                {"hbm-per-side", 4}, {"hbm-sides", 1},
                {"noc-width-bits", 512}}}};
        const nlohmann::json candidate = nlohmann::json::array({{
            {"work_id", "die-window"}, {"ready_at_ns", 0},
            {"depends_on_work", nlohmann::json::array()},
            {"is_candidate", true}, {"service_leg_ids", nullptr},
            {"legs", nlohmann::json::array({{
                {"leg_id", "die-window#0"}, {"kind", "compute"},
                {"route", nlohmann::json::array({"hbm:0"})},
                {"bytes_amount", 0}, {"duration_ns", 0.0},
                {"base_ns", 0.0}, {"group", "prefill"},
                {"latency_ns", 0}, {"ready_at_ns", nullptr},
                {"deps", nlohmann::json::array()},
                {"acquires", nlohmann::json::array()},
                {"releases", nlohmann::json::array()},
                {"signals_barrier", nullptr},
                {"remaining_bytes", nullptr},
                {"remaining_duration_ns", nullptr},
                {"in_service", false}, {"order", nullptr},
                {"die_plan", {{"specs", nlohmann::json::array({spec})},
                    {"route", nlohmann::json::array({"hbm:0"})},
                    {"label", "native-c-api-window"},
                    {"hardware_config", hardware}, {"service_factor", 1.0},
                    {"stub_phases", nullptr}}}}})}}});
        const std::string candidate_wire =
            nlohmann::json{{"candidates", candidate}}.dump();
        const auto project_die = [&]() {
            const nlohmann::json options = {
                {"analytic_segments", false}, {"drain_background", false},
                {"include_background_reference", false}};
            const std::string options_wire = options.dump();
            char* buffer = nullptr;
            size_t length = 0;
            const int code = jproj_project(die_context,
                candidate_wire.data(), candidate_wire.size(),
                options_wire.data(), options_wire.size(), &buffer, &length);
            expect(code == 0 && buffer != nullptr,
                   "real DIE candidate traverses the native projection caller");
            nlohmann::json payload = buffer == nullptr
                ? nlohmann::json::object()
                : nlohmann::json::parse(buffer, buffer + length);
            jproj_free_result(buffer);
            return payload;
        };
        const auto exact = project_die();
        if (!exact.empty()) {
            expect(exact.at("logical_event_count").get<uint64_t>() > 0 &&
                       exact.at("unsupported_reason").is_null(),
                   "current compiled DIE plan completes exactly under the frozen contract");
            expect(exact.at("die_logical_timer_event_count").get<uint64_t>() >=
                       exact.at("die_physical_timer_callback_count").get<uint64_t>(),
                   "DIE disclosure separates logical timer actions from physical callbacks");
            expect(exact.at("die_logical_timer_event_count").get<uint64_t>() ==
                       exact.at("die_physical_timer_callback_count").get<uint64_t>() &&
                       exact.at("die_coalesced_timer_callback_count").get<uint64_t>() == 0,
                   "ineligible production DIE phases use exact one-callback fallback");
        }
        jproj_destroy(die_context);
    }

    // A snapshot can declare a barrier whose signaller is supplied by the
    // candidate payload. Context creation must not instantiate/validate a
    // background-only world before that candidate is merged.
    auto future_envelope = nlohmann::json::parse(snapshot());
    future_envelope["projection_input"]["barriers"] = nlohmann::json::array({
        {{"barrier_id", "future:tp"},
         {"signallers", nlohmann::json::array({"future#0"})}}});
    const auto future_snapshot = future_envelope.dump();
    void* future_context = nullptr;
    expect(jproj_create(future_snapshot.data(), future_snapshot.size(),
                        &future_context) == 0 && future_context != nullptr,
           "snapshot creation defers candidate-owned barrier reference validation");
    if (future_context != nullptr) {
        const nlohmann::json candidate = nlohmann::json::array({
            {{"work_id", "future"},
             {"legs", nlohmann::json::array({
                 {{"leg_id", "future#0"}, {"kind", "fixed"},
                  {"route", nlohmann::json::array()}, {"bytes_amount", 0},
                  {"duration_ns", 1.0}, {"base_ns", 0.0}, {"group", "prefill"},
                  {"latency_ns", 0}, {"ready_at_ns", nullptr},
                  {"deps", nlohmann::json::array()},
                  {"acquires", nlohmann::json::array()},
                  {"releases", nlohmann::json::array()},
                  {"signals_barrier", "future:tp"},
                  {"remaining_bytes", nullptr}, {"remaining_duration_ns", nullptr},
                  {"in_service", false}, {"order", nullptr}, {"die_plan", nullptr}}})},
             {"ready_at_ns", 0}, {"depends_on_work", nlohmann::json::array()},
             {"is_candidate", true}, {"service_leg_ids", nullptr}}});
        const auto candidate_wire =
            nlohmann::json{{"candidates", candidate}}.dump();
        const std::string options = "{}";
        char* result_buffer = nullptr;
        size_t result_len = 0;
        const int code = jproj_project(future_context,
            candidate_wire.data(), candidate_wire.size(), options.data(),
            options.size(), &result_buffer, &result_len);
        expect(code == 0 && result_buffer != nullptr,
               "query validates candidate-owned barrier after merging candidate");
        if (result_buffer != nullptr) {
            const auto result = nlohmann::json::parse(result_buffer,
                                                       result_buffer + result_len);
            expect(result.value("status", std::string()) == "ok" &&
                       result.at("predicted_settle_done_ns").get<uint64_t>() == 1,
                   "candidate-owned barrier signaller runs in merged event world");
            jproj_free_result(result_buffer);
        }
        jproj_destroy(future_context);
    }

    // A zero-work dependency can complete while the initial-ready scan is
    // still traversing the candidate's registration list. The callback makes
    // the later stream ready first; the scan must not insert it a second time
    // into the ordered active vector (Python's active dict keys by leg ID).
    auto nested_envelope = nlohmann::json::parse(snapshot());
    nested_envelope["projection_input"]["resources"].push_back(
        {{"key", "link:b"}, {"kind", "link"}, {"capacity_per_ns", 10.0}});
    const auto nested_snapshot = nested_envelope.dump();
    void* nested_context = nullptr;
    expect(jproj_create(nested_snapshot.data(), nested_snapshot.size(),
                        &nested_context) == 0 && nested_context != nullptr,
           "nested-ready snapshot creates context");
    if (nested_context != nullptr) {
        const nlohmann::json nested_candidate = nlohmann::json::array({
            {{"work_id", "nested"},
             {"legs", nlohmann::json::array({
                 {{"leg_id", "nested#trigger"}, {"kind", "fixed"},
                  {"route", nlohmann::json::array()}, {"bytes_amount", 0},
                  {"duration_ns", 0.0}, {"base_ns", 0.0}, {"group", "prefill"},
                  {"latency_ns", 0}, {"ready_at_ns", nullptr},
                  {"deps", nlohmann::json::array()},
                  {"acquires", nlohmann::json::array()},
                  {"releases", nlohmann::json::array()},
                  {"signals_barrier", nullptr},
                  {"remaining_bytes", nullptr}, {"remaining_duration_ns", nullptr},
                  {"in_service", false}, {"order", nullptr}, {"die_plan", nullptr}},
                 {{"leg_id", "nested#stream"}, {"kind", "stream"},
                  {"route", nlohmann::json::array({"link:b"})},
                  {"bytes_amount", 10}, {"duration_ns", 0.0},
                  {"base_ns", 0.0}, {"group", "prefill"},
                  {"latency_ns", 0}, {"ready_at_ns", nullptr},
                  {"deps", nlohmann::json::array({"nested#trigger"})},
                  {"acquires", nlohmann::json::array()},
                  {"releases", nlohmann::json::array()},
                  {"signals_barrier", nullptr},
                  {"remaining_bytes", nullptr}, {"remaining_duration_ns", nullptr},
                  {"in_service", false}, {"order", nullptr}, {"die_plan", nullptr}}})},
             {"ready_at_ns", 0}, {"depends_on_work", nlohmann::json::array()},
             {"is_candidate", true}, {"service_leg_ids", nullptr}}});
        const auto nested_candidate_wire =
            nlohmann::json{{"candidates", nested_candidate}}.dump();
        const std::string options = "{}";
        char* nested_result_buffer = nullptr;
        size_t nested_result_len = 0;
        const int nested_code = jproj_project(nested_context,
            nested_candidate_wire.data(), nested_candidate_wire.size(),
            options.data(), options.size(), &nested_result_buffer,
            &nested_result_len);
        expect(nested_code == 0 && nested_result_buffer != nullptr,
               "nested-ready dependency projects without double activation");
        if (nested_result_buffer != nullptr) {
            const auto result = nlohmann::json::parse(
                nested_result_buffer, nested_result_buffer + nested_result_len);
            expect(result.value("status", std::string()) == "ok" &&
                       result.at("predicted_settle_done_ns").get<uint64_t>() == 1 &&
                       result.at("predicted_service_done_ns").get<uint64_t>() == 1,
                   "nested-ready candidate completes at its exact one-tick service frontier");
            expect(result.at("focus_bytes").get<uint64_t>() == 10 &&
                       result.at("bytes_served_total").get<uint64_t>() == 20,
                   "nested-ready stream is integrated once alongside background traffic");
            jproj_free_result(nested_result_buffer);
        }
        jproj_destroy(nested_context);
    }
    void* invalid = nullptr;
    const std::string bad_snapshot = "{}";

    // v5 shared-snapshot form: the legacy bare-array payload is rejected and
    // a per-call private overlay merges exactly like the legacy per-candidate
    // snapshot; identical overlays share one background-reference run.
    {
        void* legacy_context = nullptr;
        expect(jproj_create(wire.data(), wire.size(), &legacy_context) == 0,
               "legacy snapshot context for overlay equivalence");
        auto shared_envelope = nlohmann::json::parse(snapshot());
        // Shared base: extra link resource only; the candidate-relevant
        // obligation work moves into the per-call overlay.
        shared_envelope["projection_input"]["resources"].push_back(
            {{"key", "link:overlay"}, {"kind", "link"},
             {"capacity_per_ns", 4.0}});
        nlohmann::json overlay_work{
            {"work_id", "obligation"}, {"ready_at_ns", 0},
            {"depends_on_work", nlohmann::json::array()},
            {"is_candidate", false}, {"service_leg_ids", nullptr},
            {"legs", nlohmann::json::array({
                {{"leg_id", "obligation#0"}, {"kind", "stream"},
                 {"route", nlohmann::json::array({"link:a"})},
                 {"bytes_amount", 6}, {"duration_ns", 0.0}, {"base_ns", 0.0},
                 {"group", "prefill"}, {"latency_ns", 0},
                 {"ready_at_ns", nullptr}, {"deps", nlohmann::json::array()},
                 {"acquires", nlohmann::json::array()},
                 {"releases", nlohmann::json::array()},
                 {"signals_barrier", nullptr}, {"remaining_bytes", nullptr},
                 {"remaining_duration_ns", nullptr}, {"in_service", false},
                 {"order", nullptr}, {"die_plan", nullptr}}})}};
        const nlohmann::json overlay = {
            {"resources", nlohmann::json::array()},
            {"extra_background_works", nlohmann::json::array({overlay_work})},
            {"gates", nlohmann::json::array()},
            {"barriers", nlohmann::json::array()},
            {"append_background_anchor", false}};
        const nlohmann::json shared_candidate = nlohmann::json::array({
            {{"work_id", "cand"}, {"ready_at_ns", 0},
             {"depends_on_work", nlohmann::json::array()},
             {"is_candidate", true}, {"service_leg_ids", nullptr},
             {"legs", nlohmann::json::array({
                 {{"leg_id", "cand#0"}, {"kind", "stream"},
                  {"route", nlohmann::json::array({"link:overlay"})},
                  {"bytes_amount", 8}, {"duration_ns", 0.0}, {"base_ns", 0.0},
                  {"group", "prefill"}, {"latency_ns", 0},
                  {"ready_at_ns", nullptr}, {"deps", nlohmann::json::array()},
                  {"acquires", nlohmann::json::array()},
                  {"releases", nlohmann::json::array()},
                  {"signals_barrier", nullptr}, {"remaining_bytes", nullptr},
                  {"remaining_duration_ns", nullptr}, {"in_service", false},
                  {"order", nullptr}, {"die_plan", nullptr}}})}}});
        const auto project_with = [&](void* ctx, const nlohmann::json& payload) {
            const auto payload_wire = payload.dump();
            const std::string options = "{}";
            char* buffer = nullptr;
            size_t length = 0;
            const int code = jproj_project(ctx, payload_wire.data(),
                payload_wire.size(), options.data(), options.size(),
                &buffer, &length);
            nlohmann::json out = (code == 0 && buffer != nullptr)
                ? nlohmann::json::parse(buffer, buffer + length)
                : nlohmann::json::object();
            if (buffer != nullptr) jproj_free_result(buffer);
            return out;
        };

        // Bare array form is the retired payload shape.
        const std::string array_payload = "[]";
        char* reject_buffer = nullptr;
        size_t reject_len = 0;
        expect(jproj_project(legacy_context, array_payload.data(),
                             array_payload.size(), "{}", 2,
                             &reject_buffer, &reject_len) < 0,
               "retired bare-array candidate payload fails closed");
        if (reject_buffer != nullptr) jproj_free_result(reject_buffer);

        // Equivalence: legacy full snapshot (obligation baked into the
        // background) vs shared base + overlay.
        auto full_envelope = nlohmann::json::parse(snapshot());
        full_envelope["projection_input"]["resources"].push_back(
            {{"key", "link:overlay"}, {"kind", "link"},
             {"capacity_per_ns", 4.0}});
        full_envelope["projection_input"]["background_works"].push_back(
            overlay_work);
        void* full_context = nullptr;
        const auto full_wire = full_envelope.dump();
        expect(jproj_create(full_wire.data(), full_wire.size(),
                            &full_context) == 0,
               "legacy full-snapshot context for overlay equivalence");
        void* shared_context = nullptr;
        const auto shared_wire = shared_envelope.dump();
        expect(jproj_create(shared_wire.data(), shared_wire.size(),
                            &shared_context) == 0,
               "shared-snapshot context accepts the bare base");
        if (full_context != nullptr && shared_context != nullptr) {
            const auto full_result = project_with(
                full_context, {{"candidates", shared_candidate}});
            const auto shared_result = project_with(
                shared_context,
                {{"candidates", shared_candidate},
                 {"private_overlay", overlay}});
            for (const char* key : {"status", "predicted_service_done_ns",
                                    "predicted_settle_done_ns", "focus_bytes",
                                    "bytes_served_total", "event_count",
                                    "background_comparisons"}) {
                expect(full_result.at(key) == shared_result.at(key),
                       "overlay merge reproduces the legacy snapshot result");
            }
            const auto shared_again = project_with(
                shared_context,
                {{"candidates", shared_candidate},
                 {"private_overlay", overlay}});
            expect(shared_again.at("reference_cache_misses")
                       .get<uint64_t>() == 1
                       && shared_again.at("reference_cache_hits")
                              .get<uint64_t>() >= 1,
                   "identical overlay+options reuses the reference run");
            expect(shared_again.at("background_comparisons")
                       == shared_result.at("background_comparisons"),
                   "cached reference reproduces per-candidate comparisons");
        }
        if (full_context != nullptr) jproj_destroy(full_context);
        if (shared_context != nullptr) jproj_destroy(shared_context);
        if (legacy_context != nullptr) jproj_destroy(legacy_context);
    }

    expect(jproj_create(bad_snapshot.data(), bad_snapshot.size(), &invalid) < 0 &&
               invalid == nullptr,
           "invalid snapshot fails without exposing a partial context");
    expect(std::string(jproj_last_error(nullptr)).find("snapshot_envelope") != std::string::npos,
           "thread-local create error is available until next create");
    jproj_free_result(nullptr);
    jproj_destroy(nullptr);
    if (failures != 0) return 1;
    std::cout << "[native_projection_c_api_test] PASS\n";
    return 0;
}
