#include "astra-sim/workload/execution_driven/NativeProjection.hh"
#include "astra-sim/workload/execution_driven/NativeCoarseDieDomain.hh"
#include "coarse_die/Execution.hh"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

json capacity_event(std::uint64_t time_ns, std::size_t resource_index,
                    double absolute_physical_rate,
                    const std::string& resource_key = "hbm:0") {
    return {{"time_ns", time_ns}, {"resource_key", resource_key},
        {"resource_index", resource_index},
        {"absolute_physical_rate", absolute_physical_rate}};
}

json cancel_event(std::uint64_t time_ns, const std::string& leg_id) {
    return {{"time_ns", time_ns}, {"leg_id", leg_id}};
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

json envelope(bool with_endpoint, bool chain = false,
              bool overlap_endpoint = false) {
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
        // Sharing fixtures must arrive after the real startup wait. Keep
        // their original 50 ns offset from service readiness; early-arrival
        // snapshot/cancel fixtures retain their explicit absolute clock.
        const auto ready = overlap_endpoint
            ? coarse_die::compile(hardware, task())->launch_ns + 50 : 50;
        works.push_back(work("restore", ep, ready));
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

json snapshot_envelope() {
    auto wire = envelope(true, true);
    auto& input = wire["projection_input"];
    input["gates"] = json::array({{{"gate_id", "compute-slot:0"},
        {"kind", "slot"}, {"capacity_units", 1}, {"initial_occupied", 0}}});
    input["barriers"] = json::array({{{"barrier_id", "coarse-done"},
        {"signallers", json::array({"comp#0", "second#0"})}}});
    const json lease{{"gate_id", "compute-slot:0"}, {"units", 1}};
    auto& first = input["background_works"][0]["legs"][0];
    first["acquires"] = json::array({lease});
    first["releases"] = json::array({lease});
    first["signals_barrier"] = "coarse-done";

    auto second = input["background_works"][0];
    second["work_id"] = "second";
    second["legs"][0]["leg_id"] = "second#0";
    second["legs"][0]["acquires"] = json::array({lease});
    second["legs"][0]["releases"] = json::array({lease});
    second["legs"][0]["signals_barrier"] = "coarse-done";
    input["background_works"].push_back(std::move(second));

    auto timer = leg("timer#0", "fixed");
    timer["route"] = json::array();
    timer["duration_ns"] = 1;
    timer["ready_at_ns"] = 0;
    input["background_works"].push_back(work("timer", timer));

    auto joined = leg("joined#0", "fixed");
    joined["route"] = json::array();
    joined["duration_ns"] = 7;
    joined["deps"] = json::array({"coarse-done"});
    input["background_works"].push_back(work("joined", joined));
    input["focus_work_id"] = "joined";
    return wire;
}

json two_gate_recheck_envelope() {
    auto wire = envelope(false, false);
    auto& input = wire["projection_input"];
    input["gates"] = json::array({
        {{"gate_id", "gate:0"}, {"kind", "slot"}, {"capacity_units", 1},
            {"initial_occupied", 0}},
        {{"gate_id", "gate:1"}, {"kind", "slot"}, {"capacity_units", 1},
            {"initial_occupied", 0}}});
    const json lease0{{"gate_id", "gate:0"}, {"units", 1}};
    const json lease1{{"gate_id", "gate:1"}, {"units", 1}};

    auto& first = input["background_works"][0]["legs"][0];
    first["acquires"] = json::array({lease0});
    first["releases"] = json::array({lease0});

    auto second = input["background_works"][0];
    second["work_id"] = "holder1";
    second["legs"][0]["leg_id"] = "holder1#0";
    second["legs"][0]["die_plan"]["specs"] = json::array({task(), task()});
    second["legs"][0]["acquires"] = json::array({lease1});
    second["legs"][0]["releases"] = json::array({lease1});
    input["background_works"].push_back(std::move(second));

    for (const auto& name : {std::string("waiter1"), std::string("waiter2")}) {
        auto waiting = leg(name + "#0", "fixed");
        waiting["route"] = json::array();
        waiting["duration_ns"] = 7;
        waiting["acquires"] = json::array({lease0, lease1});
        waiting["releases"] = json::array({lease0, lease1});
        input["background_works"].push_back(work(name, waiting));
    }
    input["focus_work_id"] = "waiter2";
    return wire;
}

json zero_service_child_envelope() {
    auto wire = envelope(false, false);
    auto& input = wire["projection_input"];
    auto child = input["background_works"][0];
    child["work_id"] = "child";
    child["legs"][0]["leg_id"] = "child#0";
    child["legs"][0]["deps"] = json::array({"zero#0"});

    auto zero = leg("zero#0", "fixed");
    zero["route"] = json::array();
    zero["duration_ns"] = 0;
    input["background_works"] = json::array({work("zero", zero), child});
    input["focus_work_id"] = "child";
    return wire;
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
    const auto endpoint_time = with_endpoint
        ? wire["projection_input"]["background_works"][1]["ready_at_ns"]
            .get<std::uint64_t>() : 0;
    Reference result;
    for (int budget = 0; budget < 512; ++budget) {
        auto next = runtime.next_event();
        if (!arrival && (!next || *next >= endpoint_time)) next = endpoint_time;
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
        if (!arrival && *next == endpoint_time) {
            ep = runtime.submit_endpoint(endpoint(), {0, 2, 1, 1}, endpoint_time);
            arrival = true;
        }
    }
    require(result.comp != 0 && (!with_endpoint || result.restore != 0),
        "direct canonical Runtime finishes all known work");
    result.counters = runtime.counters();
    return result;
}

struct DomainReference {
    std::uint64_t comp = 0;
    std::uint64_t restore = 0;
    std::uint64_t first_plan_finish = 0;
    json counters;
    json snapshot;
};

DomainReference direct_domain(const json& wire, bool with_endpoint,
                              std::optional<std::uint64_t> final_advance_ns =
                                  std::nullopt) {
    const auto& input = wire["projection_input"];
    const auto& hardware_wire = input["coarse_die_domains"][0]["hardware_config"];
    auto hardware = coarse_die::resolve_hardware(hardware_wire);
    NativeCoarseDieDomain domain("hbm:0", hardware);
    const auto& comp_leg = input["background_works"][0]["legs"][0];
    std::vector<std::shared_ptr<const coarse_die::Plan>> plans;
    for (const auto& spec : comp_leg["die_plan"]["specs"])
        plans.push_back(coarse_die::compile(hardware, spec));
    domain.begin_plan_chain("comp#0", std::move(plans), {0, 1, 1, 1},
        input["now_ns"].get<std::uint64_t>());

    struct OrderedCapacity { std::uint64_t time; std::size_t order; json wire; };
    std::vector<OrderedCapacity> capacities;
    if (input.contains("coarse_capacity_events")) {
        for (std::size_t i = 0; i < input["coarse_capacity_events"].size(); ++i) {
            const auto& event = input["coarse_capacity_events"][i];
            capacities.push_back({event["time_ns"].get<std::uint64_t>(), i, event});
        }
        std::stable_sort(capacities.begin(), capacities.end(),
            [](const OrderedCapacity& a, const OrderedCapacity& b) {
                return a.time < b.time;
            });
    }

    std::uint64_t endpoint_time = 0;
    bool endpoint_started = !with_endpoint;
    if (with_endpoint) {
        endpoint_time = input["background_works"][1]["ready_at_ns"]
            .get<std::uint64_t>();
    }
    std::size_t capacity_index = 0;
    DomainReference result;
    auto collect = [&]() {
        for (const auto& completion : domain.take_completions()) {
            if (completion.leg_id == "comp#0") result.comp = completion.finish_ns;
            else if (completion.leg_id == "restore#0")
                result.restore = completion.finish_ns;
        }
        const auto progress = domain.progress("comp#0");
        if (result.first_plan_finish == 0 &&
            !progress["completed_plan_progress"].empty()) {
            result.first_plan_finish = progress["completed_plan_progress"][0]
                ["finish_ns"].get<std::uint64_t>();
        }
    };

    while (true) {
        std::optional<std::uint64_t> next = domain.next_event();
        if (!endpoint_started && (!next || endpoint_time < *next))
            next = endpoint_time;
        if (capacity_index < capacities.size() &&
            (!next || capacities[capacity_index].time < *next))
            next = capacities[capacity_index].time;
        if (!next) break;

        domain.advance_to(*next);
        collect();
        if (!endpoint_started && endpoint_time == *next) {
            const auto& endpoint_leg = input["background_works"][1]["legs"][0];
            domain.begin_endpoint("restore#0", endpoint_leg["coarse_endpoint"],
                {0, 2, 1, 1}, *next);
            endpoint_started = true;
            collect();
        }
        while (capacity_index < capacities.size() &&
               capacities[capacity_index].time == *next) {
            const auto& event = capacities[capacity_index].wire;
            domain.update_capacity(event["resource_index"].get<std::size_t>(),
                event["absolute_physical_rate"].get<double>(), *next);
            ++capacity_index;
            collect();
        }
    }
    if (final_advance_ns) {
        domain.advance_to(*final_advance_ns);
        collect();
    }
    collect();
    require(result.comp != 0 && (!with_endpoint || result.restore != 0),
        "direct NativeCoarseDieDomain finishes the frozen typed work");
    result.counters = domain.counters();
    result.snapshot = domain.export_snapshot();
    return result;
}

struct CancelReference {
    json progress;
    json counters;
    json snapshot;
};

CancelReference direct_cancel_endpoint(const json& wire,
                                       std::uint64_t ready_ns,
                                       std::uint64_t cancel_ns) {
    const auto& input = wire["projection_input"];
    const auto hardware = coarse_die::resolve_hardware(
        input["coarse_die_domains"][0]["hardware_config"]);
    NativeCoarseDieDomain domain("hbm:0", hardware);
    const auto& compute = input["background_works"][0]["legs"][0];
    std::vector<std::shared_ptr<const coarse_die::Plan>> plans;
    for (const auto& spec : compute["die_plan"]["specs"])
        plans.push_back(coarse_die::compile(hardware, spec));
    domain.begin_plan_chain("comp#0", std::move(plans), {0, 1, 1, 1}, 0);
    domain.advance_to(ready_ns);
    const auto& endpoint_leg = input["background_works"][1]["legs"][0];
    domain.begin_endpoint("restore#0", endpoint_leg["coarse_endpoint"],
                          {0, 2, 1, 1}, ready_ns);
    domain.advance_to(cancel_ns);
    require(domain.cancel("restore#0", cancel_ns),
        "direct canonical Domain cancels the live endpoint");
    const auto completions = domain.take_completions();
    require(completions.size() == 1 && completions.front().leg_id == "restore#0" &&
            completions.front().cancelled &&
            completions.front().finish_ns == cancel_ns,
        "direct canonical Domain emits exactly one cancelled endpoint terminal");
    return {domain.progress("restore#0"), domain.counters(),
            domain.export_snapshot()};
}

double served_hbm_bytes(const json& matrix) {
    double total = 0.0;
    for (const auto& row : matrix)
        for (const auto& direction : row)
            total += direction.get<double>();
    return total;
}

void parity(bool with_endpoint, bool chain) {
    const auto wire = envelope(with_endpoint, chain, true);
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

std::pair<std::uint64_t, std::uint64_t> direct_candidate_with_capacity(
        const json& wire, const json& candidate) {
    const auto& input = wire["projection_input"];
    const auto hardware = coarse_die::resolve_hardware(
        input["coarse_die_domains"][0]["hardware_config"]);
    const auto& background = input["background_works"][0]["legs"][0];
    const auto& candidate_leg = candidate["legs"][0];
    const auto background_plan = coarse_die::compile(hardware,
        background["die_plan"]["specs"][0]);
    const auto candidate_plan = coarse_die::compile(hardware,
        candidate_leg["die_plan"]["specs"][0]);
    coarse_die::Runtime runtime(hardware);
    const auto background_handle = runtime.submit(background_plan,
        {0, 1, 1, 1}, input["now_ns"].get<std::uint64_t>());
    const auto candidate_time = candidate["ready_at_ns"].get<std::uint64_t>();
    bool candidate_submitted = false;
    std::optional<std::uint64_t> candidate_handle;

    struct OrderedCapacity { std::uint64_t time; std::size_t order; json wire; };
    std::vector<OrderedCapacity> capacities;
    for (std::size_t i = 0; i < input["coarse_capacity_events"].size(); ++i) {
        const auto& event = input["coarse_capacity_events"][i];
        capacities.push_back({event["time_ns"].get<std::uint64_t>(), i, event});
    }
    std::stable_sort(capacities.begin(), capacities.end(),
        [](const OrderedCapacity& a, const OrderedCapacity& b) {
            return a.time < b.time;
        });
    std::size_t capacity_index = 0;
    while (!candidate_submitted || capacity_index < capacities.size()) {
        std::optional<std::uint64_t> next;
        if (!candidate_submitted) next = candidate_time;
        if (capacity_index < capacities.size() &&
            (!next || capacities[capacity_index].time < *next))
            next = capacities[capacity_index].time;
        if (!next) break;
        runtime.advance_to(*next);
        if (!candidate_submitted && candidate_time == *next) {
            candidate_handle = runtime.submit(candidate_plan, {0, 2, 1, 1}, *next);
            candidate_submitted = true;
        }
        while (capacity_index < capacities.size() &&
               capacities[capacity_index].time == *next) {
            const auto& event = capacities[capacity_index].wire;
            runtime.update_capacity(event["resource_index"].get<std::size_t>(),
                event["absolute_physical_rate"].get<double>(), *next);
            ++capacity_index;
        }
    }
    while (const auto next = runtime.next_event()) runtime.advance_to(*next);
    require(candidate_handle.has_value(), "direct candidate is admitted at its frozen ready tick");
    return {*runtime.terminal_finish_ns(background_handle),
            *runtime.terminal_finish_ns(*candidate_handle)};
}

template<class Mutation> void rejected(Mutation mutate, const char* message);
template<class Mutation> void rejected_snapshot_mutation(
        const std::shared_ptr<const NativeProjectionContext>& context,
        const json& snapshot, Mutation mutate, const char* message);

void native_known_capacity_events() {
    auto wire = envelope(true, true, true);
    const auto baseline = direct_domain(wire, true);
    const auto endpoint_time = wire["projection_input"]["background_works"][1]
        ["ready_at_ns"].get<std::uint64_t>();
    require(baseline.first_plan_finish > endpoint_time &&
        baseline.comp > baseline.first_plan_finish + 8,
        "fixture exposes an intermediate Plan-chain capacity boundary");
    const auto span = baseline.comp - baseline.first_plan_finish;
    const auto pause_delta = span / 4;
    const auto restore_delta = (span * 3) / 4;
    require(pause_delta > 0 && restore_delta > pause_delta && restore_delta < span,
        "Plan-chain fixture has room for a pause and a later capacity restore");

    const auto hardware = coarse_die::resolve_hardware(hardware_config());
    const auto hbm0 = hardware.rates[coarse_die::HBM0];
    const auto hbm1 = hardware.rates[coarse_die::HBM1];
    auto& input = wire["projection_input"];
    // Deliberately put later timestamps first. Same-tick HBM1 changes retain
    // their input order and occur after the ready endpoint has entered the
    // shared Domain.
    input["coarse_capacity_events"] = json::array({
        capacity_event(endpoint_time, coarse_die::HBM1, hbm1 * 0.5),
        capacity_event(endpoint_time, coarse_die::HBM1, hbm1),
        capacity_event(baseline.first_plan_finish + restore_delta,
            coarse_die::HBM0, hbm0),
        capacity_event(baseline.first_plan_finish,
            coarse_die::HBM0, hbm0 * 0.5),
        capacity_event(baseline.first_plan_finish + pause_delta,
            coarse_die::HBM0, 0.0)});
    const auto reference = direct_domain(wire, true);
    require(reference.comp > baseline.comp && reference.restore != 0,
        "capacity pause and restore retime the real shared Plan/endpoint Domain");

    const auto context = NativeProjectionContext::parse(wire);
    NativeProjectionWorldSession world(context);
    const auto finished = world.operation({{"operation", "run"},
        {"drain_background", true}});
    require(finished["status"] == "ok" &&
        finished["work_done_at"]["comp"] == reference.comp &&
        finished["work_done_at"]["restore"] == reference.restore,
        "world reaches the direct Domain finish ticks through the frozen event queue");
    require(finished["coarse_die_domains"]["hbm:0"]["counters"] ==
            reference.counters,
        "served/busy counters match the independent direct Domain oracle");
    require(finished["coarse_die_domains"]["hbm:0"]["counters"]["hbm_shared_ns"]
                .get<double>() > 0,
        "capacity fixture records actual endpoint/COMP shared HBM service");
    const auto snapshot = world.operation({{"operation", "export_snapshot"}});
    require(snapshot["coarse_die_domains"]["hbm:0"] == reference.snapshot,
        "complete canonical Domain snapshot matches the independent event replay");

    const auto projected = NativeProjection::project(*context,
        {{"candidates", json::array()}}, NativeProjectionOptions{});
    require(projected["predicted_settle_done_ns"] == reference.comp &&
        projected["coarse_runtime_counters_by_domain"]["hbm:0"]["counters"] ==
            reference.counters,
        "query projection inherits the same immutable known-world capacity events");

    auto terminal_wire = envelope(false, true);
    const auto terminal_baseline = direct_domain(terminal_wire, false);
    terminal_wire["projection_input"]["coarse_capacity_events"] = json::array({
        capacity_event(terminal_baseline.comp, coarse_die::HBM0, hbm0 * 0.5)});
    auto dependent = leg("dependent#0", "fixed");
    dependent["route"] = json::array();
    dependent["duration_ns"] = 7;
    dependent["deps"] = json::array({"comp#0"});
    terminal_wire["projection_input"]["background_works"].push_back(
        work("dependent", dependent));
    terminal_wire["projection_input"]["focus_work_id"] = "dependent";
    const auto terminal_reference = direct_domain(terminal_wire, false,
        terminal_baseline.comp + 7);
    NativeProjectionWorldSession terminal_world(
        NativeProjectionContext::parse(terminal_wire));
    const auto terminal_result = terminal_world.operation({{"operation", "run"}});
    require(terminal_result["status"] == "ok" &&
        terminal_result["work_done_at"]["comp"] == terminal_reference.comp &&
        terminal_result["work_done_at"]["dependent"] ==
            terminal_reference.comp + 7 &&
        terminal_world.operation({{"operation", "export_snapshot"}})
            ["coarse_die_domains"]["hbm:0"] == terminal_reference.snapshot,
        "same-tick terminal and capacity update settle the outer dependent exactly once");

    auto candidate_wire = envelope(false, false);
    auto& candidate_input = candidate_wire["projection_input"];
    candidate_input["coarse_capacity_events"] = json::array({
        capacity_event(1000, coarse_die::HBM0, hbm0),
        capacity_event(50, coarse_die::HBM0, hbm0 * 0.5)});
    const auto candidate_context = NativeProjectionContext::parse(candidate_wire);
    auto candidate = candidate_input["background_works"][0];
    candidate["work_id"] = "candidate";
    candidate["is_candidate"] = true;
    candidate["ready_at_ns"] = 50;
    candidate["legs"][0]["leg_id"] = "candidate#0";
    const auto direct_times = direct_candidate_with_capacity(candidate_wire, candidate);
    const auto candidate_result = NativeProjection::project(*candidate_context,
        {{"candidates", json::array({candidate})}}, NativeProjectionOptions{});
    require(candidate_result["status"] == "ok" &&
        candidate_result["predicted_settle_done_ns"] == direct_times.second &&
        candidate_result["background_comparisons"][0]["done_with_candidate_ns"] ==
            direct_times.first,
        "candidate overlay competes under the identical frozen capacity schedule");
}

void native_known_cancel_events() {
    const auto hardware = coarse_die::resolve_hardware(hardware_config());
    const auto hbm0 = hardware.rates[coarse_die::HBM0];

    // Cancellation at the exact ready tick has priority over endpoint
    // admission, while same-tick capacity changes still apply afterward.
    auto pending_wire = envelope(true, true);
    auto& pending_input = pending_wire["projection_input"];
    pending_input["coarse_cancel_events"] = json::array({
        cancel_event(50, "restore#0")});
    pending_input["coarse_capacity_events"] = json::array({
        capacity_event(50, coarse_die::HBM0, hbm0 * 0.5)});
    auto dependent = leg("cancel-dependent#0", "fixed");
    dependent["route"] = json::array();
    dependent["duration_ns"] = 7;
    dependent["deps"] = json::array({"restore#0"});
    pending_input["background_works"].push_back(
        work("cancel-dependent", dependent));
    pending_input["focus_work_id"] = "cancel-dependent";
    const auto pending_reference = direct_domain(pending_wire, false);
    const auto pending_context = NativeProjectionContext::parse(pending_wire);
    NativeProjectionWorldSession pending_world(pending_context);
    const auto pending_result = pending_world.operation({{"operation", "run"}});
    require(pending_result["status"] != "ok" &&
        pending_result["work_done_at"]["comp"] == pending_reference.comp &&
        pending_result["work_done_at"]["restore"].is_null() &&
        pending_result["work_done_at"]["cancel-dependent"].is_null(),
        "pending cancellation blocks its dependent without publishing success");
    require(pending_result["legs"]["restore#0"]["state"] == "cancelled" &&
        pending_result["legs"]["restore#0"]["cancelled_at_ns"] == 50 &&
        pending_result["legs"]["restore#0"]["coarse_cancelled"] == false,
        "same-tick cancel wins before ready and records an outer-only terminal");
    const auto pending_snapshot = pending_world.operation(
        {{"operation", "export_snapshot"}});
    bool pending_binding_found = false;
    for (const auto& entry : pending_snapshot["coarse_die_domains"]["hbm:0"]["entries"])
        pending_binding_found |= entry["leg_id"] == "restore#0";
    require(!pending_binding_found &&
        pending_snapshot["coarse_die_domains"]["hbm:0"] ==
            pending_reference.snapshot &&
        pending_snapshot["state"]["cancel_events_processed"] == 1 &&
        pending_snapshot["state"]["capacity_events_processed"] == 1,
        "pre-issue cancellation creates no Domain binding and capacity follows it");
    rejected_snapshot_mutation(pending_context, pending_snapshot,
        [](json& invalid) {
            auto& leg = invalid["state"]["legs"]["restore#0"];
            leg["state"] = "pending";
            leg["cancelled_at_ns"] = nullptr;
            leg["coarse_cancelled"] = false;
        }, "consumed cancellation cannot restore its target as pending");

    // An active endpoint is cancelled only after the Runtime has served a
    // real prefix. The duplicate frozen event is a no-op but remains in the
    // snapshot queue until its own ordered prefix is consumed.
    auto active_wire = envelope(true, true);
    active_wire["projection_input"]["coarse_cancel_events"] = json::array({
        cancel_event(100, "restore#0"),
        cancel_event(100, "restore#0")});
    const auto active_reference = direct_cancel_endpoint(active_wire, 50, 100);
    const double partial_bytes = served_hbm_bytes(
        active_reference.progress["last_progress"]["hbm_served_bytes"]);
    require(partial_bytes > 0.0 && partial_bytes < 1048577.0,
        "direct endpoint oracle retains a strictly partial actual-byte prefix");

    const auto active_context = NativeProjectionContext::parse(active_wire);
    NativeProjectionWorldSession active_world(active_context);
    json active_snapshot;
    bool first_cancel_consumed = false;
    for (std::uint64_t attempt = 0; attempt < 2048; ++attempt) {
        const auto before = active_world.operation(
            {{"operation", "export_snapshot"}});
        const auto consumed = before["state"]["events_processed"]
            .get<std::uint64_t>();
        (void)active_world.operation({{"operation", "run"},
            {"horizon_ns", 100}, {"event_budget", consumed + 1}});
        active_snapshot = active_world.operation(
            {{"operation", "export_snapshot"}});
        if (active_snapshot["state"]["cancel_events_processed"] == 1) {
            first_cancel_consumed = true;
            break;
        }
    }
    require(first_cancel_consumed && active_snapshot["state"]["now_ns"] == 100,
        "event budget can stop after exactly the first same-tick cancel prefix");
    std::size_t pending_duplicates = 0;
    for (const auto& event : active_snapshot["state"]["events"])
        if (event["kind"] == 5 && event["task_id"] == "1")
            ++pending_duplicates;
    require(pending_duplicates == 1,
        "the unconsumed duplicate cancellation remains queued exactly once");
    const auto active_inspect = active_world.operation(
        {{"operation", "inspect"}});
    const auto& endpoint_view = active_inspect["legs"]["restore#0"];
    require(endpoint_view["state"] == "cancelled" &&
        endpoint_view["cancelled_at_ns"] == 100 &&
        endpoint_view["coarse_cancelled"] == true &&
        endpoint_view["coarse_progress"]["state"] == "cancelled" &&
        endpoint_view["coarse_progress"]["last_progress"] ==
            active_reference.progress["last_progress"] &&
        served_hbm_bytes(endpoint_view["coarse_progress"]["last_progress"]
            ["hbm_served_bytes"]) == partial_bytes,
        "active cancellation retains canonical Domain terminal and actual service");
    require(active_snapshot["coarse_die_domains"]["hbm:0"] ==
            active_reference.snapshot,
        "active native cancellation matches the independent direct Domain replay");

    rejected_snapshot_mutation(active_context, active_snapshot,
        [](json& invalid) {
            invalid["state"].erase("cancel_events_processed");
        }, "cancel-enabled snapshots require their consumed frozen prefix");
    rejected_snapshot_mutation(active_context, active_snapshot,
        [](json& invalid) {
            invalid["state"]["cancel_events_processed"] = 0;
        }, "cancel prefix cannot omit an already consumed request");
    rejected_snapshot_mutation(active_context, active_snapshot,
        [](json& invalid) {
            invalid["state"]["cancel_events_processed"] = 2;
        }, "cancel prefix cannot consume a queued duplicate");
    rejected_snapshot_mutation(active_context, active_snapshot,
        [](json& invalid) {
            auto& events = invalid["state"]["events"];
            events.erase(std::remove_if(events.begin(), events.end(),
                [](const json& event) { return event["kind"] == 5; }),
                events.end());
        }, "snapshot cannot omit a pending frozen cancel suffix");
    rejected_snapshot_mutation(active_context, active_snapshot,
        [](json& invalid) {
            invalid["state"]["legs"]["restore#0"]["cancelled_at_ns"] = 99;
        }, "outer cancellation clock must match its consumed frozen event");
    rejected_snapshot_mutation(active_context, active_snapshot,
        [](json& invalid) {
            bool changed = false;
            for (auto& entry : invalid["coarse_die_domains"]["hbm:0"]["entries"])
                if (entry["leg_id"] == "restore#0") {
                    entry["cancelled"] = false;
                    changed = true;
                }
            require(changed, "restore endpoint binding exists for terminal mutation");
        }, "outer and retained Domain cancellation terminals must agree");
    rejected_snapshot_mutation(active_context, active_snapshot,
        [](json& invalid) {
            json pending = nullptr;
            for (const auto& event : invalid["state"]["events"])
                if (event["kind"] == 5) { pending = event; break; }
            require(!pending.is_null(), "cancel suffix exists for duplication test");
            invalid["state"]["events"].push_back(pending);
        }, "snapshot cannot duplicate a pending frozen cancellation event");

    NativeProjectionWorldSession restored(active_context);
    require(restored.operation({{"operation", "import_snapshot"},
            {"snapshot", active_snapshot}}) == active_inspect,
        "cancelled active endpoint and pending duplicate restore exactly");
    require(active_world.operation({{"operation", "run"}}) ==
            restored.operation({{"operation", "run"}}) &&
        active_world.operation({{"operation", "inspect"}}) ==
            restored.operation({{"operation", "inspect"}}) &&
        active_world.operation({{"operation", "export_snapshot"}}) ==
            restored.operation({{"operation", "export_snapshot"}}),
        "source continuation and imported fork consume the duplicate and finish identically");

    auto late_terminal_wire = envelope(true, true);
    late_terminal_wire["projection_input"]["coarse_cancel_events"] = json::array({
        cancel_event(100, "restore#0")});
    late_terminal_wire["projection_input"]["coarse_capacity_events"] = json::array({
        capacity_event(101, coarse_die::HBM0, hbm0 * 0.5)});
    const auto late_terminal_context =
        NativeProjectionContext::parse(late_terminal_wire);
    NativeProjectionWorldSession late_terminal_world(late_terminal_context);
    (void)late_terminal_world.operation({{"operation", "run"}});
    const auto late_terminal_snapshot = late_terminal_world.operation(
        {{"operation", "export_snapshot"}});
    require(late_terminal_snapshot["state"]["now_ns"] >= 101 &&
        late_terminal_snapshot["state"]["cancel_events_processed"] == 1,
        "late-terminal mutation fixture has consumed cancel and advanced beyond it");
    rejected_snapshot_mutation(late_terminal_context, late_terminal_snapshot,
        [](json& invalid) {
            auto& leg = invalid["state"]["legs"]["restore#0"];
            leg["state"] = "done";
            leg["completion_ns"] = 101;
            leg["cancelled_at_ns"] = nullptr;
            leg["coarse_cancelled"] = false;
            leg["coarse_finished"] = true;
            leg["coarse_finish_ns"] = 101;
        }, "consumed cancellation cannot be replaced by a later success terminal");

    // A true completion is settled before a cancellation request at that
    // exact tick; its dependency is released once and no cancelled state is
    // manufactured.
    auto terminal_wire = envelope(false, true);
    const auto terminal_baseline = direct_domain(terminal_wire, false);
    terminal_wire["projection_input"]["coarse_cancel_events"] = json::array({
        cancel_event(terminal_baseline.comp, "comp#0")});
    terminal_wire["projection_input"]["coarse_capacity_events"] = json::array({
        capacity_event(terminal_baseline.comp, coarse_die::HBM0, hbm0 * 0.5)});
    auto terminal_child = leg("terminal-child#0", "fixed");
    terminal_child["route"] = json::array();
    terminal_child["duration_ns"] = 7;
    terminal_child["deps"] = json::array({"comp#0"});
    terminal_wire["projection_input"]["background_works"].push_back(
        work("terminal-child", terminal_child));
    terminal_wire["projection_input"]["focus_work_id"] = "terminal-child";
    const auto terminal_reference = direct_domain(terminal_wire, false,
        terminal_baseline.comp + 7);
    const auto terminal_context = NativeProjectionContext::parse(terminal_wire);
    NativeProjectionWorldSession terminal_world(terminal_context);
    const auto terminal_result = terminal_world.operation(
        {{"operation", "run"}});
    const auto terminal_snapshot = terminal_world.operation(
        {{"operation", "export_snapshot"}});
    require(terminal_result["status"] == "ok" &&
        terminal_result["work_done_at"]["comp"] == terminal_reference.comp &&
        terminal_result["work_done_at"]["terminal-child"] ==
            terminal_reference.comp + 7 &&
        terminal_result["legs"]["comp#0"]["state"] == "done" &&
        terminal_result["legs"]["comp#0"]["cancelled_at_ns"].is_null() &&
        terminal_snapshot["coarse_die_domains"]["hbm:0"] ==
            terminal_reference.snapshot &&
        terminal_snapshot["state"]["cancel_events_processed"] == 1,
        "same-tick Runtime terminal wins before cancel and releases dependent once");
    NativeProjectionWorldSession terminal_restored(terminal_context);
    require(terminal_restored.operation({{"operation", "import_snapshot"},
            {"snapshot", terminal_snapshot}}) == terminal_result,
        "restore accepts consumed cancel whose target naturally completed at that tick");
}

void cancel_input_validation() {
    rejected([](json& input) {
        input["coarse_cancel_events"] = json::array({cancel_event(1, "missing#0")});
    }, "unknown cancellation target is rejected before world construction");
    rejected([](json& input) {
        input["coarse_cancel_events"] = json::array({cancel_event(0, "comp#0")});
        input["now_ns"] = 2;
    }, "cancellation before frozen initial clock is rejected");
    rejected([](json& input) {
        input["coarse_cancel_events"] = json::array({
            cancel_event(std::uint64_t{1} << 53, "comp#0")});
    }, "cancellation beyond exact JSON nanoseconds is rejected");
    rejected([](json& input) {
        input["background_works"][0]["legs"][0]["acquires"] = json::array({
            {{"gate_id", "slot"}, {"units", 1}}});
        input["coarse_cancel_events"] = json::array({cancel_event(0, "comp#0")});
    }, "cancellation refuses targets with ambiguous acquired gate leases");
    rejected([](json& input) {
        input["background_works"][0]["legs"][0]["releases"] = json::array({
            {{"gate_id", "slot"}, {"units", 1}}});
        input["coarse_cancel_events"] = json::array({cancel_event(0, "comp#0")});
    }, "cancellation refuses targets with gate release effects");
    rejected([](json& input) {
        input["background_works"][0]["legs"][0]["signals_barrier"] = "done";
        input["barriers"] = json::array({{{"barrier_id", "done"},
            {"signallers", json::array({"comp#0"})}}});
        input["coarse_cancel_events"] = json::array({cancel_event(0, "comp#0")});
    }, "cancellation refuses a target that would otherwise signal a barrier");
    rejected([](json& input) {
        input["barriers"] = json::array({{{"barrier_id", "done"},
            {"signallers", json::array({"comp#0"})}}});
        input["coarse_cancel_events"] = json::array({cancel_event(0, "comp#0")});
    }, "barrier registry signallers are also protected from cancellation");
    rejected([](json& input) {
        auto event = cancel_event(0, "comp#0");
        event["unexpected"] = true;
        input["coarse_cancel_events"] = json::array({event});
    }, "unknown cancellation-event fields are rejected");
}

void capacity_input_validation() {
    const auto hardware = coarse_die::resolve_hardware(hardware_config());
    const auto valid = capacity_event(0, coarse_die::HBM0,
        hardware.rates[coarse_die::HBM0]);
    rejected([&](json& input) {
        input["coarse_capacity_events"] = json::array({capacity_event(0,
            coarse_die::HBM0, hardware.rates[coarse_die::HBM0], "hbm:missing")});
    }, "unknown capacity-event domain is rejected before world construction");
    rejected([&](json& input) {
        auto event = valid; event["resource_index"] = coarse_die::resource_count;
        input["coarse_capacity_events"] = json::array({event});
    }, "capacity resource index outside canonical hardware is rejected");
    rejected([&](json& input) {
        auto event = valid; event["absolute_physical_rate"] = -1.0;
        input["coarse_capacity_events"] = json::array({event});
    }, "negative physical rate is rejected before execution");
    rejected([&](json& input) {
        auto event = valid;
        event["absolute_physical_rate"] = std::numeric_limits<double>::quiet_NaN();
        input["coarse_capacity_events"] = json::array({event});
    }, "NaN physical rate is rejected before execution");
    rejected([&](json& input) {
        auto event = valid;
        event["absolute_physical_rate"] = std::numeric_limits<double>::infinity();
        input["coarse_capacity_events"] = json::array({event});
    }, "infinite physical rate is rejected before execution");
    rejected([&](json& input) {
        auto event = valid;
        event["absolute_physical_rate"] = hardware.rates[coarse_die::HBM0] + 1.0;
        input["coarse_capacity_events"] = json::array({event});
    }, "physical rate above the resolved hardware peak is rejected");
    rejected([&](json& input) {
        auto event = valid; event["time_ns"] = 0.5;
        input["coarse_capacity_events"] = json::array({event});
    }, "fractional capacity event time is rejected");
    rejected([&](json& input) {
        auto event = valid; event["time_ns"] = std::uint64_t{1} << 53;
        input["coarse_capacity_events"] = json::array({event});
    }, "capacity event beyond exact JSON nanoseconds is rejected");
    rejected([&](json& input) {
        input["now_ns"] = 10;
        auto event = valid; event["time_ns"] = 9;
        input["coarse_capacity_events"] = json::array({event});
    }, "capacity event before frozen initial clock is rejected");
    rejected([](json& input) {
        input["coarse_capacity_events"] = json::array({
            {{"time_ns", 0}, {"resource_key", "hbm:0"},
             {"resource_index", coarse_die::HBM0}}});
    }, "incomplete capacity-event object is rejected");
    rejected([](json& input) {
        auto event = capacity_event(0, coarse_die::HBM0, 1.0);
        event["unexpected"] = true;
        input["coarse_capacity_events"] = json::array({event});
    }, "unknown capacity-event keys are rejected");
    require(hardware.rates[coarse_die::INJECT0] == 0.0,
        "canonical fixture exposes a disabled resource for capacity preflight");
    rejected([](json& input) {
        input["coarse_capacity_events"] = json::array({
            capacity_event(0, coarse_die::INJECT0, 1.0)});
    }, "disabled physical resource can only remain at zero rate");
}

void capacity_budget_snapshot_prefix() {
    auto wire = envelope(true, true);
    const auto hardware = coarse_die::resolve_hardware(hardware_config());
    wire["projection_input"]["coarse_capacity_events"] = json::array({
        capacity_event(0, coarse_die::HBM0, 0.0),
        capacity_event(0, coarse_die::HBM0,
            hardware.rates[coarse_die::HBM0])});
    const auto context = NativeProjectionContext::parse(wire);
    NativeProjectionWorldSession source(context);
    const auto interrupted = source.operation({{"operation", "run"},
        {"event_budget", 1}});
    require(interrupted["status"] == "degraded" &&
        interrupted["now_ns"] == 0 && interrupted["events_processed"] == 1,
        "event budget stops after a consumed same-tick capacity prefix");
    const auto checkpoint = source.operation({{"operation", "export_snapshot"}});
    require(checkpoint["state"]["capacity_events_processed"] == 1,
        "snapshot records the exact consumed capacity prefix");
    std::uint64_t same_tick_suffix = 0;
    for (const auto& event : checkpoint["state"]["events"])
        if (event["kind"] == 4 && event["time_ns"] == 0) ++same_tick_suffix;
    require(same_tick_suffix == 1,
        "unprocessed capacity suffix remains queued at its consumed tick");

    rejected_snapshot_mutation(context, checkpoint,
        [](json& invalid) {
            invalid["state"].erase("capacity_events_processed");
        }, "nonempty frozen capacity schedule requires its consumed-prefix state");
    rejected_snapshot_mutation(context, checkpoint,
        [](json& invalid) {
            invalid["state"]["capacity_events_processed"] = 0;
        }, "capacity prefix cannot omit an already consumed same-tick event");
    rejected_snapshot_mutation(context, checkpoint,
        [](json& invalid) {
            invalid["state"]["capacity_events_processed"] = 2;
        }, "capacity prefix cannot consume an event still in the queue suffix");
    rejected_snapshot_mutation(context, checkpoint,
        [](json& invalid) {
            auto& events = invalid["state"]["events"];
            events.erase(std::remove_if(events.begin(), events.end(),
                [](const json& event) { return event["kind"] == 4; }),
                events.end());
        }, "snapshot cannot omit the pending same-tick capacity suffix");
    rejected_snapshot_mutation(context, checkpoint,
        [](json& invalid) {
            json pending = nullptr;
            for (const auto& event : invalid["state"]["events"])
                if (event["kind"] == 4) { pending = event; break; }
            require(!pending.is_null(), "capacity suffix fixture has a pending event");
            invalid["state"]["events"].push_back(pending);
        }, "snapshot cannot duplicate a pending frozen capacity event");

    NativeProjectionWorldSession restored(context);
    const auto restored_inspect = restored.operation({{"operation", "import_snapshot"},
        {"snapshot", checkpoint}});
    require(restored_inspect == source.operation({{"operation", "inspect"}}),
        "same-tick capacity checkpoint imports without advancing its source");
    const auto source_done = source.operation({{"operation", "run"},
        {"drain_background", true}});
    const auto restored_done = restored.operation({{"operation", "run"},
        {"drain_background", true}});
    require(source_done == restored_done &&
        source.operation({{"operation", "inspect"}}) ==
            restored.operation({{"operation", "inspect"}}) &&
        source.operation({{"operation", "export_snapshot"}}) ==
            restored.operation({{"operation", "export_snapshot"}}),
        "resumed and uninterrupted worlds drain to identical full state");
    const auto reference = direct_domain(wire, true);
    require(source_done["work_done_at"]["comp"] == reference.comp &&
        source_done["work_done_at"]["restore"] == reference.restore &&
        source_done["coarse_die_domains"]["hbm:0"]["counters"] ==
            reference.counters,
        "resumed same-tick pause uses the later restore and matches direct Domain execution");

    auto combined_wire = envelope(false, false);
    combined_wire["projection_input"]["forecast_arrival_placeholders_ns"] =
        json::array({0});
    combined_wire["projection_input"]["coarse_capacity_events"] = json::array({
        capacity_event(0, coarse_die::HBM0,
            hardware.rates[coarse_die::HBM0] * 0.5)});
    const auto combined_context = NativeProjectionContext::parse(combined_wire);
    NativeProjectionWorldSession combined(combined_context);
    const auto two_events = combined.operation({{"operation", "run"},
        {"event_budget", 2}});
    const auto combined_snapshot = combined.operation({{"operation", "export_snapshot"}});
    require(two_events["events_processed"] == 2 &&
        combined_snapshot["state"]["forecast_events_processed"] == 1 &&
        combined_snapshot["state"]["capacity_events_processed"] == 1,
        "forecast and capacity queues consume one event each at the same tick");
    rejected_snapshot_mutation(combined_context, combined_snapshot,
        [](json& invalid) {
            invalid["state"]["events_processed"] = 1;
        }, "combined frozen prefixes cannot exceed the total event counter");
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

void require_rejected_import_preserves(
        NativeProjectionWorldSession& session, const json& invalid,
        const json& original_inspect, const json& original_snapshot,
        const std::string& label) {
    bool threw = false;
    try {
        (void)session.operation({{"operation", "import_snapshot"},
            {"snapshot", invalid}});
    } catch (const std::exception&) { threw = true; }
    require(threw, label.c_str());
    require(session.operation({{"operation", "inspect"}}) == original_inspect,
        "invalid snapshot import leaves the original world inspect unchanged");
    require(session.operation({{"operation", "export_snapshot"}}) == original_snapshot,
        "invalid snapshot import leaves the original full snapshot unchanged");
}

template<class Mutation> void rejected_snapshot_mutation(
        const std::shared_ptr<const NativeProjectionContext>& context,
        const json& snapshot, Mutation mutate, const char* message) {
    NativeProjectionWorldSession receiver(context);
    const auto original_inspect = receiver.operation({{"operation", "inspect"}});
    const auto original_snapshot = receiver.operation({{"operation", "export_snapshot"}});
    auto invalid = snapshot;
    mutate(invalid);
    require_rejected_import_preserves(receiver, invalid, original_inspect,
        original_snapshot, message);
}

void snapshot_round_trip(
        const std::shared_ptr<const NativeProjectionContext>& context,
        NativeProjectionWorldSession& source,
        const json& frozen_input,
        bool exercise_invalid_imports = false,
        bool expect_fixed_timer_to_settle_next_step = false) {
    const auto source_inspect = source.operation({{"operation", "inspect"}});
    const auto exported = source.operation({{"operation", "export_snapshot"}});
    const auto snapshot = json::parse(exported.dump());
    require(snapshot == exported,
        "world snapshot survives JSON dump/parse without wire changes");
    require(snapshot["schema"] == "native_coarse_projection_world_snapshot_v1" &&
        snapshot["snapshot_version"] == 1,
        "world snapshot advertises the supported versioned schema");
    require(snapshot["frozen_input"] == frozen_input,
        "snapshot binds the exact frozen input wire");

    NativeProjectionWorldSession imported(context);
    const auto original_inspect = imported.operation({{"operation", "inspect"}});
    const auto original_snapshot = imported.operation({{"operation", "export_snapshot"}});
    if (exercise_invalid_imports) {
        auto invalid = snapshot;
        invalid["unexpected"] = true;
        require_rejected_import_preserves(imported, invalid, original_inspect,
            original_snapshot, "unknown top-level snapshot keys are rejected");

        invalid = snapshot;
        invalid["schema"] = "native_coarse_projection_world_snapshot_v0";
        require_rejected_import_preserves(imported, invalid, original_inspect,
            original_snapshot, "unknown world snapshot schema is rejected");

        invalid = snapshot;
        invalid["state"]["legs"]["joined#0"]["timer_scheduled"] = true;
        require_rejected_import_preserves(imported, invalid, original_inspect,
            original_snapshot, "pending dependency cannot own a scheduled ready timer");

        invalid = snapshot;
        invalid["state"]["now_ns"] = invalid["state"]["now_ns"].get<std::uint64_t>() + 1;
        require_rejected_import_preserves(imported, invalid, original_inspect,
            original_snapshot, "world and coarse-domain clock mismatch is rejected");

        invalid = snapshot;
        invalid["frozen_input"]["now_ns"] = 1;
        require_rejected_import_preserves(imported, invalid, original_inspect,
            original_snapshot, "snapshot from a different frozen input is rejected");

        invalid = snapshot;
        invalid["coarse_die_domains"]["hbm:0"]["unexpected"] = true;
        require_rejected_import_preserves(imported, invalid, original_inspect,
            original_snapshot, "unknown coarse-domain snapshot keys are rejected");

        invalid = snapshot;
        auto& owner_wire = invalid["coarse_die_domains"]["hbm:0"]["entries"][0]["owner"];
        coarse_die::Owner wrong_owner{
            owner_wire["request_id"].get<std::uint64_t>(),
            owner_wire["node_id"].get<std::uint64_t>(),
            owner_wire["execution_id"].get<std::uint64_t>() + 1000,
            owner_wire["generation"].get<std::uint64_t>()};
        owner_wire["execution_id"] = wrong_owner.execution_id;
        owner_wire["key"] = wrong_owner.key();
        require_rejected_import_preserves(imported, invalid, original_inspect,
            original_snapshot, "domain entry owned by another native leg is rejected");

        invalid = snapshot;
        invalid["kernel_contract"] = invalid["kernel_contract"].is_null()
            ? json{{"different_contract", true}} : json::object();
        require_rejected_import_preserves(imported, invalid, original_inspect,
            original_snapshot, "snapshot with a different kernel contract is rejected");
    }

    const auto restored = imported.operation({{"operation", "import_snapshot"},
        {"snapshot", snapshot}});
    require(restored == source_inspect,
        "snapshot import restores every observable native inspect field");
    require(imported.operation({{"operation", "export_snapshot"}}) == snapshot,
        "restored world re-exports the exact snapshot state");

    NativeProjectionWorldSession replay(context);
    const auto replayed = replay.operation({{"operation", "import_snapshot"},
        {"snapshot", snapshot}});
    require(replayed == source_inspect,
        "independent session imports the same checkpoint");
    NativeProjectionWorldSession independent(context);
    const auto independent_initial = independent.operation({{"operation", "inspect"}});

    const auto imported_step = imported.operation({{"operation", "step"}});
    require(replay.operation({{"operation", "inspect"}}) == replayed,
        "one restored session advances without mutating its peer");
    require(source.operation({{"operation", "inspect"}}) == source_inspect,
        "restoring and advancing a checkpoint never mutates its source world");
    require(independent.operation({{"operation", "inspect"}}) == independent_initial,
        "restored sessions do not share mutable state with a fresh session");

    const auto replay_step = replay.operation({{"operation", "step"}});
    require(imported_step == replay_step &&
        imported.operation({{"operation", "inspect"}}) ==
            replay.operation({{"operation", "inspect"}}),
        "replayed worlds have identical next-step results and complete inspect state");
    if (expect_fixed_timer_to_settle_next_step) {
        const auto after_step = imported.operation({{"operation", "inspect"}});
        require(after_step["work_done_at"]["timer"] == 1,
            "a fixed timer integrated exactly to its terminal tick settles on the next step");
    }

    const auto source_step = source.operation({{"operation", "step"}});
    require(source_step == imported_step &&
        source.operation({{"operation", "inspect"}}) ==
            imported.operation({{"operation", "inspect"}}),
        "uninterrupted source and restored world have identical next-step state");

    const auto source_run = source.operation({{"operation", "run"}, {"event_budget", 0}});
    const auto left_run = imported.operation({{"operation", "run"}, {"event_budget", 0}});
    const auto right_run = replay.operation({{"operation", "run"}, {"event_budget", 0}});
    require(source_run == left_run && left_run == right_run &&
        source.operation({{"operation", "inspect"}}) ==
            imported.operation({{"operation", "inspect"}}) &&
        imported.operation({{"operation", "inspect"}}) ==
            replay.operation({{"operation", "inspect"}}),
        "uninterrupted and restored worlds drain to exactly equal terminal inspect state");
    const auto source_restored = source.operation({{"operation", "import_snapshot"},
        {"snapshot", snapshot}});
    require(source_restored == source_inspect,
        "source session returns to its checkpoint after its direct-run oracle is captured");
    require(independent.operation({{"operation", "inspect"}}) == independent_initial,
        "draining restored worlds does not mutate an independent session");
}

const json* coarse_entry(const json& snapshot, const std::string& leg_id) {
    for (const auto& entry : snapshot["coarse_die_domains"]["hbm:0"]["entries"])
        if (entry["leg_id"] == leg_id) return &entry;
    return nullptr;
}

void snapshot_world_states() {
    const auto wire = snapshot_envelope();
    const auto context = NativeProjectionContext::parse(wire);
    NativeProjectionWorldSession world(context);
    auto inspected = world.operation({{"operation", "inspect"}});
    require(inspected["legs"]["second#0"]["state"] == "gate_wait" &&
        inspected["gates"]["compute-slot:0"]["waiting_leg_ids"] ==
            json::array({"second#0"}),
        "snapshot fixture starts with a real queued native gate waiter");
    require(inspected["barriers"]["coarse-done"]["has_passed"] == false &&
        inspected["legs"]["restore#0"]["state"] == "timer",
        "snapshot fixture starts before the barrier and known 50ns arrival");
    require(inspected["legs"]["timer#0"]["state"] == "active",
        "snapshot fixture includes a live fixed timer");
    snapshot_round_trip(context, world, wire["projection_input"], true);

    auto before_arrival = world.operation({{"operation", "run"}, {"horizon_ns", 49}});
    require(before_arrival["legs"]["restore#0"]["state"] == "timer" &&
        coarse_entry(world.operation({{"operation", "export_snapshot"}}), "restore#0") == nullptr,
        "known endpoint remains queued before its 50ns external arrival");
    snapshot_round_trip(context, world, wire["projection_input"]);

    auto after_arrival = world.operation({{"operation", "run"}, {"horizon_ns", 50}});
    const auto after_arrival_snapshot = world.operation({{"operation", "export_snapshot"}});
    require(after_arrival["now_ns"] == 50 &&
        after_arrival["legs"]["restore#0"]["state"] == "active" &&
        coarse_entry(after_arrival_snapshot, "restore#0") != nullptr,
        "known endpoint is submitted to the same native domain at 50ns");
    snapshot_round_trip(context, world, wire["projection_input"]);

    bool found_chain_boundary = false;
    for (int attempt = 0; attempt < 20000; ++attempt) {
        const auto current = world.operation({{"operation", "export_snapshot"}});
        const auto* entry = coarse_entry(current, "comp#0");
        if (entry && (*entry)["plan_index"] == 1 &&
            (*entry)["completed_plan_progress"].size() == 1) {
            found_chain_boundary = true;
            require((*entry)["completed_plan_progress"][0]["terminal"] == true,
                "chain boundary retains the completed first Plan progress");
            break;
        }
        require(world.operation({{"operation", "step"}})["now_ns"] >= 50,
            "coarse chain advances from the post-arrival checkpoint");
    }
    require(found_chain_boundary,
        "native scheduler reaches and exposes an intermediate Plan-chain boundary");
    snapshot_round_trip(context, world, wire["projection_input"]);

    bool found_partial_barrier = false;
    for (int attempt = 0; attempt < 20000; ++attempt) {
        inspected = world.operation({{"operation", "inspect"}});
        const auto& arrived = inspected["barriers"]["coarse-done"]["arrived"];
        if (arrived.size() == 1) {
            found_partial_barrier = true;
            require(inspected["barriers"]["coarse-done"]["has_passed"] == false &&
                inspected["legs"]["second#0"]["state"] == "active",
                "first coarse terminal signals a pending barrier while the second lease runs");
            break;
        }
        require(world.operation({{"operation", "step"}})["now_ns"] >= 50,
            "native scheduler advances while waiting for the first barrier signal");
    }
    require(found_partial_barrier, "barrier exposes an intermediate arrival state");
    snapshot_round_trip(context, world, wire["projection_input"]);

    bool found_passed_barrier = false;
    for (int attempt = 0; attempt < 20000; ++attempt) {
        inspected = world.operation({{"operation", "inspect"}});
        if (inspected["barriers"]["coarse-done"]["has_passed"] == true) {
            found_passed_barrier = true;
            require(inspected["barriers"]["coarse-done"]["arrived"].size() == 2 &&
                inspected["legs"]["joined#0"]["state"] == "active",
                "all coarse signallers release the dependent fixed join");
            break;
        }
        require(world.operation({{"operation", "step"}})["now_ns"] >= 50,
            "native scheduler advances to the barrier terminal state");
    }
    require(found_passed_barrier, "barrier reaches a stable passed state");
    snapshot_round_trip(context, world, wire["projection_input"]);

    const auto terminal = world.operation({{"operation", "run"}});
    require(terminal["status"] == "ok" && terminal["work_done_at"].contains("joined"),
        "rich snapshot fixture drains its endpoint, Plan chains, gate, barrier, and fixed join");
    snapshot_round_trip(context, world, wire["projection_input"]);
}

void snapshot_partial_integration_and_budget() {
    const auto wire = snapshot_envelope();
    const auto context = NativeProjectionContext::parse(wire);
    NativeProjectionWorldSession partial(context);
    const auto integrated = partial.operation({{"operation", "integrate"},
        {"target_ns", 1}});
    require(integrated["now_ns"] == 1 &&
        integrated["legs"]["timer#0"]["state"] == "active" &&
        integrated["legs"]["timer#0"]["remaining_duration_ns"] == 0 &&
        integrated["work_done_at"]["timer"].is_null(),
        "controlled integration to a fixed timer boundary leaves completion pending");
    snapshot_round_trip(context, partial, wire["projection_input"], false, true);

    NativeProjectionWorldSession budgeted(context);
    const auto interrupted = budgeted.operation({{"operation", "run"},
        {"event_budget", 1}});
    require(interrupted["status"] == "degraded" &&
        interrupted["events_processed"] >= 1,
        "small event budget leaves a restorable in-flight world");
    const auto snapshot = budgeted.operation({{"operation", "export_snapshot"}});
    require(snapshot["state"]["event_budget_limit"] == 1,
        "snapshot preserves the event-budget interruption limit");
    snapshot_round_trip(context, budgeted, wire["projection_input"]);
}

void snapshot_forecast_prefix() {
    auto wire = envelope(false, false);
    wire["projection_input"]["forecast_arrival_placeholders_ns"] =
        json::array({0, 0});
    const auto context = NativeProjectionContext::parse(wire);
    NativeProjectionWorldSession world(context);
    const auto interrupted = world.operation({{"operation", "run"},
        {"event_budget", 1}});
    require(interrupted["status"] == "degraded" &&
        interrupted["now_ns"] == 0 && interrupted["events_processed"] == 1,
        "event budget can interrupt a same-tick forecast prefix");
    const auto snapshot = world.operation({{"operation", "export_snapshot"}});
    require(snapshot["state"]["forecast_events_processed"] == 1,
        "snapshot counts the consumed forecast prefix separately");
    std::uint64_t pending_at_now = 0;
    for (const auto& event : snapshot["state"]["events"])
        if (event["kind"] == 3 && event["time_ns"] == snapshot["state"]["now_ns"])
            ++pending_at_now;
    require(pending_at_now == 1,
        "unprocessed forecast suffix may remain queued exactly at the snapshot clock");

    rejected_snapshot_mutation(context, snapshot,
        [](json& invalid) {
            invalid["state"]["forecast_events_processed"] = 0;
        }, "forecast queue must match the exact consumed-prefix count");
    rejected_snapshot_mutation(context, snapshot,
        [](json& invalid) {
            invalid["state"]["forecast_events_processed"] = 2;
        }, "snapshot cannot consume a forecast that remains in its event suffix");
    rejected_snapshot_mutation(context, snapshot,
        [](json& invalid) {
            auto& events = invalid["state"]["events"];
            events.erase(std::remove_if(events.begin(), events.end(),
                [](const json& event) { return event["kind"] == 3; }), events.end());
        }, "snapshot cannot omit an unconsumed same-tick forecast arrival");

    snapshot_round_trip(context, world, wire["projection_input"]);
    const auto drained = world.operation({{"operation", "run"}, {"event_budget", 0}});
    const auto drained_snapshot = world.operation({{"operation", "export_snapshot"}});
    require(drained["status"] == "ok" &&
        drained_snapshot["state"]["forecast_events_processed"] == 2,
        "resumed session consumes the remaining forecast suffix exactly once");
    for (const auto& event : drained_snapshot["state"]["events"])
        require(event["kind"] != 3,
            "no processed forecast remains in the restored native event queue");
}

void snapshot_two_gate_recheck_and_zero_service_activation() {
    const auto gate_wire = two_gate_recheck_envelope();
    const auto gate_context = NativeProjectionContext::parse(gate_wire);
    NativeProjectionWorldSession gate_world(gate_context);
    const auto gate_initial = gate_world.operation({{"operation", "inspect"}});
    require(gate_initial["legs"]["waiter1#0"]["state"] == "gate_wait" &&
        gate_initial["legs"]["waiter2#0"]["state"] == "gate_wait" &&
        gate_initial["gates"]["gate:0"]["waiting_leg_ids"] ==
            json::array({"waiter1#0", "waiter2#0"}),
        "multi-gate fixture preserves FIFO waiters on the first blocked gate");
    snapshot_round_trip(gate_context, gate_world, gate_wire["projection_input"]);
    const auto gate_done = gate_world.operation({{"operation", "run"}});
    require(gate_done["status"] == "ok" &&
        gate_done["work_done_at"]["comp"] < gate_done["work_done_at"]["holder1"] &&
        gate_done["work_done_at"]["holder1"] < gate_done["work_done_at"]["waiter1"] &&
        gate_done["work_done_at"]["waiter1"] < gate_done["work_done_at"]["waiter2"] &&
        gate_done["gates"]["gate:0"]["occupied"] == 0 &&
        gate_done["gates"]["gate:1"]["occupied"] == 0,
        "releasing the second coarse lease retries the first gate queue in FIFO order");

    const auto child_wire = zero_service_child_envelope();
    const auto child_context = NativeProjectionContext::parse(child_wire);
    NativeProjectionWorldSession child_world(child_context);
    const auto child_initial = child_world.operation({{"operation", "inspect"}});
    const auto child_snapshot = child_world.operation({{"operation", "export_snapshot"}});
    require(child_initial["work_done_at"]["zero"] == 0 &&
        child_initial["legs"]["child#0"]["state"] == "active" &&
        child_snapshot["coarse_die_domains"]["hbm:0"]["entries"].size() == 1 &&
        child_snapshot["coarse_die_domains"]["hbm:0"]["entries"][0]["leg_id"] == "child#0",
        "zero-service predecessor activates its coarse child exactly once during build");
    snapshot_round_trip(child_context, child_world, child_wire["projection_input"]);
    const auto child_done = child_world.operation({{"operation", "run"}});
    require(child_done["status"] == "ok" &&
        child_done["work_done_at"].contains("child"),
        "restored zero-service dependency graph reaches its native child terminal");
}

void snapshot_readiness_and_latency_events() {
    auto wire = snapshot_envelope();
    auto& input = wire["projection_input"];
    auto delayed = leg("delayed#0", "fixed");
    delayed["route"] = json::array();
    delayed["duration_ns"] = 7;
    delayed["latency_ns"] = 4;
    delayed["ready_at_ns"] = 3;
    input["background_works"].push_back(work("delayed", delayed));

    auto zero = leg("zero#0", "fixed");
    zero["route"] = json::array();
    zero["duration_ns"] = 0;
    zero["latency_ns"] = 9;
    input["background_works"].push_back(work("zero_latency", zero));

    const auto context = NativeProjectionContext::parse(wire);
    NativeProjectionWorldSession world(context);
    const auto initial = world.operation({{"operation", "inspect"}});
    const auto initial_snapshot = world.operation({{"operation", "export_snapshot"}});
    bool ready_event_pending = false;
    for (const auto& event : initial_snapshot["state"]["events"])
        ready_event_pending |= event["task_id"] == "delayed#0" &&
            event["kind"] == 2 && event["time_ns"] == 3;
    require(initial["legs"]["delayed#0"]["state"] == "timer" &&
        ready_event_pending,
        "snapshot preserves a future ready event for a delayed fixed leg");
    require(initial["legs"]["zero#0"]["state"] == "done" &&
        initial["legs"]["zero#0"]["completion_ns"] == 0 &&
        initial["work_done_at"]["zero_latency"] == 0,
        "zero-duration fixed work completes immediately even when latency is nonzero");
    rejected_snapshot_mutation(context, initial_snapshot,
        [](json& invalid) {
            auto& state = invalid["state"];
            auto& zero_leg = state["legs"]["zero#0"];
            zero_leg["state"] = "latency";
            zero_leg["completion_ns"] = nullptr;
            const auto sequence = state["sequence"].get<std::uint64_t>();
            state["sequence"] = sequence + 1;
            state["events"].push_back({{"time_ns", 9}, {"priority", 1},
                {"producer", "zero#0"}, {"sequence", sequence},
                {"kind", 1}, {"task_id", "zero#0"}});
        }, "zero-duration fixed work cannot be forged into a latency wait");
    rejected_snapshot_mutation(context, initial_snapshot,
        [](json& invalid) {
            auto& state = invalid["state"];
            auto& zero_leg = state["legs"]["zero#0"];
            zero_leg["state"] = "active";
            zero_leg["completion_ns"] = nullptr;
            state["active_leg_ids"].push_back("zero#0");
        }, "zero-duration fixed work cannot be forged into active service");
    snapshot_round_trip(context, world, wire["projection_input"]);

    const auto before_ready = world.operation({{"operation", "run"},
        {"horizon_ns", 2}});
    require(before_ready["legs"]["delayed#0"]["state"] == "timer",
        "fixed work remains pending before its ready timestamp");
    snapshot_round_trip(context, world, wire["projection_input"]);

    const auto at_ready = world.operation({{"operation", "run"},
        {"horizon_ns", 3}});
    const auto latency_snapshot = world.operation({{"operation", "export_snapshot"}});
    bool latency_event_pending = false;
    for (const auto& event : latency_snapshot["state"]["events"])
        latency_event_pending |= event["task_id"] == "delayed#0" &&
            event["kind"] == 1 && event["time_ns"] == 7;
    require(at_ready["now_ns"] == 3 &&
        at_ready["legs"]["delayed#0"]["state"] == "latency" &&
        latency_event_pending,
        "ready arrival transitions to a restorable pending latency event");
    rejected_snapshot_mutation(context, latency_snapshot,
        [](json& invalid) {
            for (auto& event : invalid["state"]["events"])
                if (event["task_id"] == "delayed#0" && event["kind"] == 1)
                    event["time_ns"] = 6;
        }, "latency event cannot imply service before exact graph readiness");
    const auto integrated_latency = world.operation({{"operation", "integrate"},
        {"target_ns", 4}});
    const auto later_latency_snapshot = world.operation(
        {{"operation", "export_snapshot"}});
    require(integrated_latency["now_ns"] == 4 &&
        integrated_latency["legs"]["delayed#0"]["state"] == "latency",
        "controlled integration can checkpoint strictly before the latency event");
    rejected_snapshot_mutation(context, later_latency_snapshot,
        [](json& invalid) {
            for (auto& event : invalid["state"]["events"])
                if (event["task_id"] == "delayed#0" && event["kind"] == 1)
                    event["time_ns"] = 8;
        }, "latency deadline must equal graph readiness plus its fixed latency");
    snapshot_round_trip(context, world, wire["projection_input"]);

    bool found_active_fixed = false;
    json active_snapshot;
    for (int attempt = 0; attempt < 2048; ++attempt) {
        const auto current = world.operation({{"operation", "inspect"}});
        if (current["legs"]["delayed#0"]["state"] == "active") {
            found_active_fixed = true;
            active_snapshot = world.operation({{"operation", "export_snapshot"}});
            break;
        }
        require(world.operation({{"operation", "step"}})["progressed"] == true,
            "ready latency event progresses to fixed active service");
    }
    require(found_active_fixed &&
        active_snapshot["state"]["legs"]["delayed#0"]["remaining_duration_ns"] == 7,
        "latency completion starts the fixed service at its exact readiness floor");
    rejected_snapshot_mutation(context, active_snapshot,
        [](json& invalid) {
            invalid["state"]["legs"]["delayed#0"]["remaining_duration_ns"] = 6;
        }, "ungated active fixed service cannot start before its readiness floor");
    const auto integrated_active = world.operation({{"operation", "integrate"},
        {"target_ns", 8}});
    const auto later_active_snapshot = world.operation(
        {{"operation", "export_snapshot"}});
    require(integrated_active["now_ns"] == 8 &&
        integrated_active["legs"]["delayed#0"]["state"] == "active" &&
        integrated_active["legs"]["delayed#0"]["remaining_duration_ns"] == 6,
        "controlled integration records exact active service progress after latency");
    rejected_snapshot_mutation(context, later_active_snapshot,
        [](json& invalid) {
            invalid["state"]["legs"]["delayed#0"]["remaining_duration_ns"] = 7;
        }, "active fixed service must start at exact graph readiness after latency");
    snapshot_round_trip(context, world, wire["projection_input"]);

    const auto terminal = world.operation({{"operation", "run"}});
    require(terminal["legs"]["delayed#0"]["state"] == "done",
        "latency-delayed fixed leg reaches a terminal snapshot state");
    const auto terminal_snapshot = world.operation({{"operation", "export_snapshot"}});
    rejected_snapshot_mutation(context, terminal_snapshot,
        [](json& invalid) {
            auto& completion = invalid["state"]["legs"]["delayed#0"]["completion_ns"];
            completion = completion.get<std::uint64_t>() + 1;
        }, "terminal fixed completion must reconcile with latency and service duration");
    snapshot_round_trip(context, world, wire["projection_input"]);
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
        native_known_capacity_events();
        native_known_cancel_events();
        cancel_input_validation();
        capacity_input_validation();
        capacity_budget_snapshot_prefix();
        native_gate_and_dependency_join();
        snapshot_world_states();
        snapshot_partial_integration_and_budget();
        snapshot_forecast_prefix();
        snapshot_two_gate_recheck_and_zero_service_activation();
        snapshot_readiness_and_latency_events();
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
