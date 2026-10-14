#include "astra-sim/workload/execution_driven/NativeProjection.hh"
#include "coarse_die/Execution.hh"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using json = nlohmann::json;
using namespace AstraSim::ExecutionDriven;

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

json hardware_config() {
    std::ifstream input(NATIVE_COARSE_CONFIG);
    require(input.good(), "tracked coarse hardware fixture exists");
    json config;
    input >> config;
    return config;
}

json task() {
    return {{"schema", "coarse_die_task_v2"}, {"op_kind", "gemm"},
        {"phase", "prefill"},
        {"tensor_shapes", {{"m", 65}, {"n", 65}, {"k", 4097}}}};
}

json endpoint() {
    return {{"endpoint_id", 17}, {"transaction_id", 17},
        {"owner_kind", "restore"}, {"direction", "write"},
        {"physical_bytes", 1048577}, {"eta_hbm", 1.0},
        {"path_class", "edge_to_hbm"},
        {"route", {{"internal_route", false}}}};
}

json leg(const std::string& id, const std::string& kind = "compute") {
    return {{"leg_id", id}, {"kind", kind}, {"route", json::array({"hbm:0"})},
        {"bytes_amount", 0}, {"duration_ns", 0}, {"base_ns", 0},
        {"group", "prefill"}, {"latency_ns", 0}, {"ready_at_ns", nullptr},
        {"deps", json::array()}, {"acquires", json::array()},
        {"releases", json::array()}, {"signals_barrier", nullptr},
        {"remaining_bytes", nullptr}, {"remaining_duration_ns", nullptr},
        {"in_service", false}, {"order", nullptr}, {"die_plan", nullptr}};
}

json work(const std::string& id, json item, std::uint64_t ready = 0) {
    return {{"work_id", id}, {"legs", json::array({std::move(item)})},
        {"ready_at_ns", ready}, {"depends_on_work", json::array()},
        {"is_candidate", false}, {"service_leg_ids", nullptr}};
}

json envelope(bool with_endpoint, bool chain = false) {
    const auto config = hardware_config();
    const auto hardware = coarse_die::resolve_hardware(config);
    auto comp = leg("comp#0");
    auto specs = json::array({task()});
    if (chain) specs.push_back({{"schema", "coarse_die_task_v2"},
        {"op_kind", "norm"}, {"phase", "prefill"},
        {"tensor_shapes", {{"m", 128}, {"hidden", 512}}}});
    comp["die_plan"] = {{"specs", specs}, {"route", comp["route"]},
        {"label", "canonical coarse chain"}, {"hardware_config", config},
        {"service_factor", 1.0}, {"stub_phases", nullptr}};
    auto works = json::array({work("comp", comp)});
    if (with_endpoint) {
        auto ep = leg("restore#0", "stream");
        ep["coarse_endpoint"] = endpoint();
        works.push_back(work("restore", ep, 50));
    }
    return {{"projection_input", {{"now_ns", 0},
        {"resources", json::array({{{"key", "hbm:0"}, {"kind", "hbm"},
            {"capacity_per_ns", (hardware.rates[coarse_die::HBM0] +
                                 hardware.rates[coarse_die::HBM1]) / 1e9}}})},
        {"background_works", works}, {"focus_work_id", "comp"},
        {"gates", json::array()}, {"barriers", json::array()},
        {"efficiency", {{"eta", json::object()}, {"gamma", json::object()}}},
        {"pipelines", json::array()},
        {"forecast_arrival_placeholders_ns", json::array()},
        {"coarse_die_domains", json::array({{{"resource_key", "hbm:0"},
            {"hardware_config", config}}})}}}};
}

struct Reference { std::uint64_t comp = 0, restore = 0; json counters; };
Reference direct(const json& wire, bool with_endpoint) {
    const auto hardware = coarse_die::resolve_hardware(hardware_config());
    coarse_die::Runtime runtime(hardware);
    const auto& specs = wire["projection_input"]["background_works"][0]
        ["legs"][0]["die_plan"]["specs"];
    std::vector<std::shared_ptr<const coarse_die::Plan>> plans;
    for (const auto& spec : specs) plans.push_back(coarse_die::compile(hardware, spec));
    std::size_t index = 0;
    auto comp = runtime.submit(plans[0], {0, 1, 1, 1}, 0);
    std::optional<std::uint64_t> ep;
    bool arrival = !with_endpoint;
    Reference result;
    for (int budget = 0; budget < 512; ++budget) {
        auto next = runtime.next_event();
        if (!arrival && (!next || *next >= 50)) next = 50;
        if (!next) break;
        runtime.advance_to(*next);
        // Complete old work before the known arrival at the same tick.
        if (!result.comp) {
            const auto finish = runtime.terminal_finish_ns(comp);
            if (finish) {
                runtime.destroy_execution(comp);
                if (++index < plans.size())
                    comp = runtime.submit(plans[index], {0, 1, 1, 1}, *finish);
                else result.comp = *finish;
            }
        }
        if (ep && !result.restore) {
            const auto finish = runtime.terminal_finish_ns(*ep);
            if (finish) {
                result.restore = *finish;
                runtime.destroy_execution(*ep);
            }
        }
        if (!arrival && *next == 50) {
            ep = runtime.submit_endpoint(endpoint(), {0, 2, 1, 1}, 50);
            arrival = true;
        }
    }
    require(result.comp != 0 && (!with_endpoint || result.restore != 0),
        "direct canonical Runtime finishes all known work");
    result.counters = runtime.counters();
    return result;
}

void parity(bool with_endpoint, bool chain) {
    const auto wire = envelope(with_endpoint, chain);
    const auto reference = direct(wire, with_endpoint);
    const auto context = NativeProjectionContext::parse(wire);
    NativeProjectionWorldSession session(context);
    const auto initial = session.operation({{"operation", "inspect"}});
    auto finished = session.operation({{"operation", "run"}});
    require(finished["status"] == "ok", "native coarse world drains");
    require(finished["work_done_at"]["comp"] == reference.comp,
        "native Plan/chain terminal tick equals direct canonical Runtime");
    if (with_endpoint)
        require(finished["work_done_at"]["restore"] == reference.restore,
            "native external local-HBM terminal equals direct canonical Runtime");
    const auto counters = finished["coarse_die_domains"]["hbm:0"]["counters"];
    require(counters == reference.counters,
        "same-world served, busy, fairness update and event counters match exactly");
    if (with_endpoint)
        require(counters["hbm_shared_ns"].get<double>() > 0,
            "fixture observes actual shared HBM service");
    require(finished["live_world_import_supported"] == false,
        "fresh-world bridge discloses live import boundary");
    const auto repeated = NativeProjection::project(*context,
        {{"candidates", json::array()}}, NativeProjectionOptions{});
    require(repeated["predicted_settle_done_ns"] == reference.comp,
        "query world uses the same coarse event bridge");
    const auto reset = session.operation({{"operation", "reset"}});
    require(reset == initial, "reset reproduces fresh private world without candidate state");
    NativeProjectionWorldSession other(context);
    require(other.operation({{"operation", "inspect"}}) == initial,
        "another native world never shares mutable coarse progress");
    const auto first_step = session.operation({{"operation", "step"}});
    require(first_step["progressed"] == true, "coarse next_event wakes the native world");
    require(other.operation({{"operation", "inspect"}}) == initial,
        "advancing one world leaves another unchanged");
}

void candidate_world_parity() {
    const auto wire = envelope(false);
    const auto context = NativeProjectionContext::parse(wire);
    auto candidate = wire["projection_input"]["background_works"][0];
    candidate["work_id"] = "candidate";
    candidate["is_candidate"] = true;
    candidate["ready_at_ns"] = 50;
    candidate["legs"][0]["leg_id"] = "candidate#0";
    const json payload{{"candidates", json::array({candidate})}};
    const auto hardware = coarse_die::resolve_hardware(hardware_config());
    const auto plan = coarse_die::compile(hardware, task());
    coarse_die::Runtime direct(hardware);
    const auto first = direct.submit(plan, {0, 1, 1, 1}, 0);
    direct.advance_to(50);
    const auto second = direct.submit(plan, {0, 2, 1, 1}, 50);
    while (const auto next = direct.next_event()) direct.advance_to(*next);
    const auto before = NativeProjection::project(*context,
        {{"candidates", json::array()}}, NativeProjectionOptions{});
    const auto result = NativeProjection::project(*context, payload,
        NativeProjectionOptions{});
    require(result["status"] == "ok" &&
        result["predicted_settle_done_ns"] == *direct.terminal_finish_ns(second),
        "candidate and known background use the canonical concurrent Runtime");
    require(result["background_comparisons"][0]["done_with_candidate_ns"] ==
        *direct.terminal_finish_ns(first), "background delay is from the same competing world");
    const auto after = NativeProjection::project(*context,
        {{"candidates", json::array()}}, NativeProjectionOptions{});
    require(before["predicted_settle_done_ns"] == after["predicted_settle_done_ns"] &&
        before["coarse_runtime_counters_by_domain"] == after["coarse_runtime_counters_by_domain"],
        "candidate does not mutate the frozen baseline or its memoized reference run");
}

void native_gate_and_dependency_join() {
    auto wire = envelope(false);
    auto& input = wire["projection_input"];
    const auto solo = direct(wire, false).comp;
    input["gates"] = json::array({{{"gate_id", "compute-slot:0"},
        {"kind", "slot"}, {"capacity_units", 1}, {"initial_occupied", 0}}});
    const json lease{{"gate_id", "compute-slot:0"}, {"units", 1}};
    auto& first = input["background_works"][0]["legs"][0];
    first["acquires"] = json::array({lease});
    first["releases"] = json::array({lease});
    auto second = input["background_works"][0];
    second["work_id"] = "second";
    second["legs"][0]["leg_id"] = "second#0";
    input["background_works"].push_back(second);
    auto joined = leg("joined#0", "fixed");
    joined["route"] = json::array();
    joined["duration_ns"] = 7;
    joined["deps"] = json::array({"comp#0", "second#0"});
    input["background_works"].push_back(work("joined", joined));
    input["focus_work_id"] = "joined";
    NativeProjectionWorldSession world(NativeProjectionContext::parse(wire));
    const auto result = world.operation({{"operation", "run"}});
    require(result["work_done_at"]["comp"] == solo &&
        result["work_done_at"]["second"] == solo * 2 &&
        result["work_done_at"]["joined"] == solo * 2 + 7,
        "coarse outer terminal releases native slot and graph dependencies exactly once");
    require(result["gates"]["compute-slot:0"]["occupied"] == 0 &&
        result["gates"]["compute-slot:0"]["max_occupied"] == 1 &&
        result["gates"]["compute-slot:0"]["waiting_leg_ids"].empty(),
        "native admission gate is balanced after two coarse executions");
}

template<class Mutation> void rejected(Mutation mutate, const char* message) {
    auto wire = envelope(true);
    mutate(wire["projection_input"]);
    bool threw = false;
    try {
        auto context = NativeProjectionContext::parse(wire);
        NativeProjectionWorldSession world(context);
        (void)world;
    } catch (const std::exception&) { threw = true; }
    require(threw, message);
}
}

int main() {
    try {
        parity(false, false);
        parity(true, false);
        parity(true, true);
        candidate_world_parity();
        native_gate_and_dependency_join();
        rejected([](json& input) { input["now_ns"] = 0.5; },
            "fractional legacy clock cannot silently round coarse service");
        rejected([](json& input) {
            input["forecast_arrival_placeholders_ns"] =
                json::array({std::uint64_t{1} << 53}); },
            "known forecast timestamps are validated before coarse work starts");
        rejected([](json& input) { input["efficiency"]["eta"]["hbm"] = 0.5; },
            "prediction-only efficiency cannot alter the shared model");
        rejected([](json& input) { input["background_works"][0]["legs"][0]
            ["die_plan"]["service_factor"] = 2; },
            "legacy service factor cannot wrap coarse Plan");
        rejected([](json& input) { auto& item = input["background_works"][1]["legs"][0];
            item.erase("coarse_endpoint"); item["bytes_amount"] = 1048577; },
            "scalar HBM bytes cannot bypass the one coarse arbiter");
        rejected([](json& input) { input["background_works"][0]["legs"][0]
            ["in_service"] = true; },
            "scalar active state cannot masquerade as complete live snapshot");
        rejected([](json& input) { input["coarse_die_domains"] = json::array(); },
            "candidate cannot invent an unfrozen physical coarse domain");
        rejected([](json& input) { input["resources"][0]["capacity_per_ns"] = 1.0; },
            "native HBM capacity must match the canonical descriptor");
        std::cout << "NativeCoarseProjectionTest ALL PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "NativeCoarseProjectionTest: " << error.what() << '\n';
        return 1;
    }
}
