/******************************************************************************
Native exact-projection input contract and world implementation.
******************************************************************************/

#include "astra-sim/workload/execution_driven/NativeProjection.hh"

#include "astra-sim/workload/execution_driven/PipelineDescriptor.hh"
#include "astra-sim/workload/execution_driven/PipelineKernel.hh"
#include "DieModel.hh"

#include <algorithm>
#include <cmath>
#include <map>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace AstraSim {
namespace ExecutionDriven {

struct NativeProjectionLease final {
    std::string gate_id;
    uint64_t units = 1;
};

struct NativeProjectionDiePhase final {
    enum class Kind { Timer, Memory } kind = Kind::Timer;
    uint64_t ns = 0;
    uint64_t stage_id = 0;
    uint64_t actual_bytes = 0;
    double service_units = 0.0;
};

struct NativeProjectionDiePlan final {
    std::vector<std::shared_ptr<const DieModel::DieOpSpec>> specs;
    std::vector<std::shared_ptr<const DieModel::DiePlan>> plans;
    std::vector<NativeProjectionDiePhase> stub_phases;
    std::vector<std::string> route;
    std::string label;
    double service_factor = 1.0;
    bool is_stub = false;
};

struct NativeProjectionLegSpec final {
    std::string leg_id;
    std::string kind;
    std::vector<std::string> route;
    uint64_t bytes_amount = 0;
    double duration_ns = 0.0;
    double base_ns = 0.0;
    std::string group;
    uint64_t latency_ns = 0;
    bool has_ready_at = false;
    uint64_t ready_at_ns = 0;
    std::vector<std::string> deps;
    std::vector<NativeProjectionLease> acquires;
    std::vector<NativeProjectionLease> releases;
    bool has_signal_barrier = false;
    std::string signals_barrier;
    bool has_remaining_bytes = false;
    uint64_t remaining_bytes = 0;
    bool has_remaining_duration = false;
    double remaining_duration_ns = 0.0;
    bool in_service = false;
    bool has_order = false;
    int64_t order = 0;
    bool has_registration_ordinal = false;
    uint64_t registration_ordinal = 0;
    std::shared_ptr<const NativeProjectionDiePlan> die_plan;
};

struct NativeProjectionWorkSpec final {
    std::string work_id;
    std::vector<NativeProjectionLegSpec> legs;
    uint64_t ready_at_ns = 0;
    std::vector<std::string> depends_on_work;
    bool is_candidate = false;
    bool has_service_leg_ids = false;
    std::vector<std::string> service_leg_ids;
    std::vector<std::shared_ptr<const PipelineDescriptor>> pipeline_descriptors;
};

struct NativeProjectionResource final {
    std::string key;
    std::string kind;
    double capacity_per_ns = 0.0;
};

struct NativeProjectionGate final {
    std::string gate_id;
    std::string kind;
    uint64_t capacity_units = 0;
    uint64_t initial_occupied = 0;
};

struct NativeProjectionBarrier final {
    std::string barrier_id;
    std::vector<std::string> signallers;
};

struct NativeProjectionBlockPipeline final {
    std::string pipeline_id;
    std::string fill_resource;
    uint64_t block_bytes = 0;
    uint64_t num_blocks = 0;
    double consume_duration_ns = 0.0;
    std::string buffer_gate_id;
    uint64_t buffer_slots = 0;
    uint64_t startup_ns = 0;
};

struct NativeProjectionInput final {
    double now_ns = 0.0;
    std::vector<NativeProjectionResource> resources;
    std::vector<NativeProjectionWorkSpec> background_works;
    // Only caller-provided WorkItems participate in the public closure and
    // comparison tables. Internal BlockPipelineSpec work is scheduled in the
    // same world but is not an explicit background ledger row.
    std::vector<std::string> explicit_background_work_ids;
    std::string focus_work_id;
    std::vector<NativeProjectionGate> gates;
    std::vector<NativeProjectionBarrier> barriers;
    std::vector<NativeProjectionBlockPipeline> pipelines;
    std::map<std::string, double> eta;
    std::map<std::string, double> gamma;
    std::vector<uint64_t> forecast_arrival_placeholders_ns;
    // Shared-snapshot form: number of leading background_works entries that
    // precede the forecast works.  A per-call private overlay splices its
    // extra background works at this index so the merged work order matches
    // the legacy per-candidate snapshot ([shared] + [extras] + [forecast]).
    // Defaults to "no forecast section" (splice at the end).
    size_t forecast_start_index = static_cast<size_t>(-1);
};

/// Per-call candidate-private additions to a shared snapshot context.
/// Mirrors exactly the candidate-dependent parts that the legacy per-candidate
/// ProjectionInput embedded at creation time: candidate-declared resources
/// (setdefault semantics), candidate gates, candidate barriers, and the §7.2
/// inherited-obligation background works.
struct NativeProjectionOverlay final {
    std::vector<NativeProjectionResource> resources;
    std::vector<NativeProjectionWorkSpec> extra_background_works;
    std::vector<NativeProjectionGate> gates;
    std::vector<NativeProjectionBarrier> barriers;
    bool append_background_anchor = false;
};

namespace {

using Json = nlohmann::json;

[[noreturn]] void fail(const std::string& path, const std::string& message) {
    throw std::invalid_argument("native projection " + path + ": " + message);
}

void object(const Json& value, const std::string& path,
            const std::set<std::string>& required,
            const std::set<std::string>& optional = {}) {
    if (!value.is_object()) fail(path, "expected object");
    for (const auto& key : required) {
        if (value.find(key) == value.end()) fail(path, "missing key '" + key + "'");
    }
    for (auto it = value.begin(); it != value.end(); ++it) {
        if (!required.count(it.key()) && !optional.count(it.key())) {
            fail(path, "unknown key '" + it.key() + "'");
        }
    }
}

uint64_t uint_value(const Json& value, const std::string& path) {
    if (value.is_number_unsigned()) return value.get<uint64_t>();
    if (!value.is_number_integer()) fail(path, "expected integer, not bool/float/string");
    const int64_t n = value.get<int64_t>();
    if (n < 0) fail(path, "must be nonnegative");
    return static_cast<uint64_t>(n);
}

double number_value(const Json& value, const std::string& path) {
    if (!value.is_number()) fail(path, "expected number");
    const double n = value.get<double>();
    if (!std::isfinite(n)) fail(path, "must be finite");
    return n;
}

bool has_prefix(const std::string& value, const char* prefix) {
    return value.compare(0, std::char_traits<char>::length(prefix), prefix) == 0;
}

std::string string_value(const Json& value, const std::string& path,
                         bool allow_empty = false) {
    if (!value.is_string()) fail(path, "expected string");
    auto s = value.get<std::string>();
    if (!allow_empty && s.empty()) fail(path, "must not be empty");
    return s;
}

void array(const Json& value, const std::string& path) {
    if (!value.is_array()) fail(path, "expected array");
}

void validate_projection_input_shape(const Json& input) {
    object(input, "projection_input",
           {"now_ns", "resources", "background_works", "focus_work_id",
            "gates", "barriers", "efficiency", "pipelines",
            "forecast_arrival_placeholders_ns"},
           {"forecast_start_index", "defer_focus_validation"});
    if (number_value(input.at("now_ns"), "projection_input.now_ns") < 0.0) {
        fail("projection_input.now_ns", "must be nonnegative");
    }
    string_value(input.at("focus_work_id"), "projection_input.focus_work_id");
    for (const char* key : {"resources", "background_works", "gates",
                            "barriers", "pipelines",
                            "forecast_arrival_placeholders_ns"}) {
        array(input.at(key), std::string("projection_input.") + key);
    }
    const auto& efficiency = input.at("efficiency");
    object(efficiency, "projection_input.efficiency", {"eta", "gamma"});
    for (const char* table_name : {"eta", "gamma"}) {
        const auto& table = efficiency.at(table_name);
        if (!table.is_object()) fail("projection_input.efficiency", "eta/gamma must be objects");
        for (auto it = table.begin(); it != table.end(); ++it) {
            string_value(Json(it.key()), "efficiency key");
            if (number_value(it.value(), "efficiency factor") <= 0.0) {
                fail("projection_input.efficiency", "factors must be positive");
            }
        }
    }

    std::set<std::string> resource_keys;
    for (size_t i = 0; i < input.at("resources").size(); ++i) {
        const auto& item = input.at("resources")[i];
        const auto path = "projection_input.resources[" + std::to_string(i) + "]";
        object(item, path, {"key", "kind", "capacity_per_ns"});
        const auto key = string_value(item.at("key"), path + ".key");
        const auto kind = string_value(item.at("kind"), path + ".kind");
        const auto capacity = number_value(item.at("capacity_per_ns"), path + ".capacity_per_ns");
        if (!resource_keys.insert(key).second || capacity <= 0.0) {
            fail(path, "duplicate resource key or nonpositive capacity");
        }
        if (kind != "link" && kind != "hbm" && kind != "pool") {
            fail(path + ".kind", "unsupported resource kind");
        }
        const std::string prefix = key.substr(0, key.find(':'));
        if (prefix != kind) fail(path, "resource key prefix disagrees with kind");
    }

    auto validate_lease = [&](const Json& lease, const std::string& path) {
        object(lease, path, {"gate_id", "units"});
        string_value(lease.at("gate_id"), path + ".gate_id");
        if (uint_value(lease.at("units"), path + ".units") == 0) {
            fail(path + ".units", "must be positive");
        }
    };
    auto validate_work = [&](const Json& work, const std::string& path,
                             int expected_candidate) {
        object(work, path, {"work_id", "legs", "ready_at_ns",
                            "depends_on_work", "is_candidate", "service_leg_ids"},
               {"pipeline_descriptors"});
        bool has_descriptors = false;
        if (work.contains("pipeline_descriptors")) {
            array(work.at("pipeline_descriptors"), path + ".pipeline_descriptors");
            has_descriptors = !work.at("pipeline_descriptors").empty();
        }
        string_value(work.at("work_id"), path + ".work_id");
        (void)uint_value(work.at("ready_at_ns"), path + ".ready_at_ns");
        if (!work.at("is_candidate").is_boolean() ||
            (expected_candidate >= 0 &&
             work.at("is_candidate").get<bool>() != (expected_candidate != 0))) {
            fail(path + ".is_candidate", "candidate flag mismatch");
        }
        array(work.at("legs"), path + ".legs");
        if (work.at("legs").empty() && !has_descriptors) {
            fail(path + ".legs", "empty legs require pipeline_descriptors");
        }
        array(work.at("depends_on_work"), path + ".depends_on_work");
        if (!work.at("service_leg_ids").is_null()) array(work.at("service_leg_ids"), path + ".service_leg_ids");
        for (size_t j = 0; j < work.at("legs").size(); ++j) {
            const auto& leg = work.at("legs")[j];
            const auto lp = path + ".legs[" + std::to_string(j) + "]";
            object(leg, lp, {"leg_id", "kind", "route", "bytes_amount",
                "duration_ns", "base_ns", "group", "latency_ns", "ready_at_ns",
                "deps", "acquires", "releases", "signals_barrier",
                "remaining_bytes", "remaining_duration_ns", "in_service",
                "order", "die_plan"}, {"registration_ordinal"});
            string_value(leg.at("leg_id"), lp + ".leg_id");
            const auto kind = string_value(leg.at("kind"), lp + ".kind");
            if (kind != "stream" && kind != "fixed" && kind != "compute") {
                fail(lp + ".kind", "must be stream/fixed/compute");
            }
            if (!leg.at("route").is_array()) fail(lp + ".route", "expected array");
            for (const auto& route : leg.at("route")) {
                const auto key = string_value(route, lp + ".route[]");
                if (!resource_keys.count(key)) fail(lp + ".route", "unknown resource '" + key + "'");
            }
            (void)uint_value(leg.at("bytes_amount"), lp + ".bytes_amount");
            (void)uint_value(leg.at("latency_ns"), lp + ".latency_ns");
            if (number_value(leg.at("duration_ns"), lp + ".duration_ns") < 0.0 ||
                number_value(leg.at("base_ns"), lp + ".base_ns") < 0.0) {
                fail(lp, "duration/base must be nonnegative");
            }
            string_value(leg.at("group"), lp + ".group", true);
            for (const char* key : {"deps", "acquires", "releases"}) {
                if (!leg.at(key).is_array()) fail(lp + "." + key, "expected array");
            }
            for (const auto& dep : leg.at("deps")) string_value(dep, lp + ".deps[]");
            for (const auto& lease : leg.at("acquires")) validate_lease(lease, lp + ".acquires[]");
            for (const auto& lease : leg.at("releases")) validate_lease(lease, lp + ".releases[]");
            if (!leg.at("signals_barrier").is_null()) string_value(leg.at("signals_barrier"), lp + ".signals_barrier");
            if (!leg.at("remaining_bytes").is_null()) (void)uint_value(leg.at("remaining_bytes"), lp + ".remaining_bytes");
            if (!leg.at("remaining_duration_ns").is_null() &&
                number_value(leg.at("remaining_duration_ns"), lp + ".remaining_duration_ns") < 0.0) {
                fail(lp + ".remaining_duration_ns", "must be nonnegative");
            }
            if (!leg.at("in_service").is_boolean()) fail(lp + ".in_service", "expected boolean");
            if (!leg.at("order").is_null()) {
                const auto& order = leg.at("order");
                if (order.is_number_unsigned()) {
                    if (order.get<uint64_t>() >
                        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                        fail(lp + ".order", "exceeds int64 completion-order domain");
                    }
                } else if (!order.is_number_integer()) {
                    fail(lp + ".order", "expected signed integer or null");
                }
            }
            if (!leg.at("die_plan").is_null() && !leg.at("die_plan").is_object()) {
                fail(lp + ".die_plan", "must be a typed die-plan object or null");
            }
            if (leg.contains("registration_ordinal") &&
                !leg.at("registration_ordinal").is_null()) {
                (void)uint_value(leg.at("registration_ordinal"),
                                 lp + ".registration_ordinal");
            }
            if (has_descriptors &&
                (!leg.contains("registration_ordinal") ||
                 leg.at("registration_ordinal").is_null())) {
                fail(lp + ".registration_ordinal",
                     "descriptor-bearing work requires ordinary-leg registration ordinals");
            }
        }
    };
    std::set<std::string> work_ids;
    for (size_t i = 0; i < input.at("background_works").size(); ++i) {
        const auto& work = input.at("background_works")[i];
        const auto path = "projection_input.background_works[" + std::to_string(i) + "]";
        validate_work(work, path, -1);
        if (!work_ids.insert(work.at("work_id").get<std::string>()).second) {
            fail(path + ".work_id", "duplicate work id");
        }
    }
    // Shared-snapshot form defers focus validation to overlay application:
    // the shared base may be empty (the anchor decision depends on the
    // per-call overlay's extras), so the focus entry is a placeholder that
    // apply_projection_overlay recomputes from the merged background.
    const bool defer_focus = input.contains("defer_focus_validation")
        && input.at("defer_focus_validation").is_boolean()
        && input.at("defer_focus_validation").get<bool>();
    if (!defer_focus
            && !work_ids.count(input.at("focus_work_id").get<std::string>())) {
        fail("projection_input.focus_work_id", "not in background_works");
    }

    std::set<std::string> gate_ids;
    for (size_t i = 0; i < input.at("gates").size(); ++i) {
        const auto& gate = input.at("gates")[i];
        const auto path = "projection_input.gates[" + std::to_string(i) + "]";
        object(gate, path, {"gate_id", "kind", "capacity_units", "initial_occupied"});
        auto id = string_value(gate.at("gate_id"), path + ".gate_id");
        const auto kind = string_value(gate.at("kind"), path + ".kind");
        auto cap = uint_value(gate.at("capacity_units"), path + ".capacity_units");
        auto occ = uint_value(gate.at("initial_occupied"), path + ".initial_occupied");
        if (!gate_ids.insert(id).second || cap == 0 || occ > cap ||
            (kind != "buffer" && kind != "capacity" && kind != "slot")) {
            fail(path, "invalid/duplicate gate");
        }
    }
    for (size_t i = 0; i < input.at("barriers").size(); ++i) {
        const auto& barrier = input.at("barriers")[i];
        const auto path = "projection_input.barriers[" + std::to_string(i) + "]";
        object(barrier, path, {"barrier_id", "signallers"});
        string_value(barrier.at("barrier_id"), path + ".barrier_id");
        array(barrier.at("signallers"), path + ".signallers");
        if (barrier.at("signallers").empty()) fail(path + ".signallers", "must not be empty");
        for (const auto& id : barrier.at("signallers")) string_value(id, path + ".signallers[]");
    }
    for (size_t i = 0; i < input.at("pipelines").size(); ++i) {
        const auto& pipeline = input.at("pipelines")[i];
        const auto path = "projection_input.pipelines[" + std::to_string(i) + "]";
        object(pipeline, path, {"pipeline_id", "fill_resource", "block_bytes",
            "num_blocks", "consume_duration_ns", "buffer_gate_id",
            "buffer_slots", "startup_ns"});
        const auto pipeline_id = string_value(pipeline.at("pipeline_id"), path + ".pipeline_id");
        const auto fill_resource = string_value(pipeline.at("fill_resource"), path + ".fill_resource");
        const auto gate_id = string_value(pipeline.at("buffer_gate_id"), path + ".buffer_gate_id");
        const auto block_bytes = uint_value(pipeline.at("block_bytes"), path + ".block_bytes");
        const auto num_blocks = uint_value(pipeline.at("num_blocks"), path + ".num_blocks");
        const auto consume_duration = number_value(
            pipeline.at("consume_duration_ns"), path + ".consume_duration_ns");
        const auto slots = uint_value(pipeline.at("buffer_slots"), path + ".buffer_slots");
        (void)uint_value(pipeline.at("startup_ns"), path + ".startup_ns");
        if (!resource_keys.count(fill_resource)) {
            fail(path + ".fill_resource", "unknown resource");
        }
        if (!gate_ids.count(gate_id)) {
            fail(path + ".buffer_gate_id", "unknown gate");
        }
        if (block_bytes == 0 || num_blocks == 0 || slots == 0 ||
            !std::isfinite(consume_duration) || consume_duration < 0.0) {
            fail(path, "block_bytes/num_blocks/buffer_slots must be positive and consume duration nonnegative");
        }
        if (num_blocks > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / 2)) {
            fail(path + ".num_blocks", "expansion exceeds addressable task count");
        }
        const auto gate = std::find_if(input.at("gates").begin(), input.at("gates").end(),
            [&](const Json& item) { return item.at("gate_id").get<std::string>() == gate_id; });
        if (gate == input.at("gates").end() ||
            slots > uint_value(gate->at("capacity_units"), path + ".buffer_slots")) {
            fail(path + ".buffer_slots", "exceeds the declared buffer gate capacity");
        }
        const std::string synthetic_work = "pipeline:" + pipeline_id;
        if (!work_ids.insert(synthetic_work).second) {
            fail(path + ".pipeline_id", "duplicates a registered work ID");
        }
    }
    for (const auto& tick : input.at("forecast_arrival_placeholders_ns")) {
        if (number_value(tick, "forecast arrival tick") < number_value(input.at("now_ns"), "now_ns")) {
            fail("projection_input.forecast_arrival_placeholders_ns", "ticks must be >= now_ns");
        }
    }
}

int64_t int64_value(const Json& value, const std::string& path) {
    if (value.is_number_unsigned()) {
        const uint64_t n = value.get<uint64_t>();
        if (n > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            fail(path, "integer exceeds int64 completion-order domain");
        }
        return static_cast<int64_t>(n);
    }
    if (!value.is_number_integer()) fail(path, "expected signed integer or null");
    return value.get<int64_t>();
}

std::vector<std::string> parse_string_array(const Json& value,
                                            const std::string& path) {
    array(value, path);
    std::vector<std::string> result;
    result.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        result.push_back(string_value(value[i], path + "[" + std::to_string(i) + "]"));
    }
    return result;
}

std::vector<NativeProjectionLease> parse_leases(
    const Json& value, const std::string& path) {
    array(value, path);
    std::vector<NativeProjectionLease> result;
    result.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        const auto lease_path = path + "[" + std::to_string(i) + "]";
        const auto& lease = value[i];
        object(lease, lease_path, {"gate_id", "units"});
        result.push_back({string_value(lease.at("gate_id"), lease_path + ".gate_id"),
                          uint_value(lease.at("units"), lease_path + ".units")});
    }
    return result;
}

std::shared_ptr<const NativeProjectionDiePlan> parse_die_plan(
    const Json& wire, const std::string& path) {
    object(wire, path, {"specs", "route", "label", "hardware_config",
                        "service_factor", "stub_phases"});
    auto result = std::make_shared<NativeProjectionDiePlan>();
    array(wire.at("specs"), path + ".specs");
    result->route = parse_string_array(wire.at("route"), path + ".route");
    if (result->route.empty()) fail(path + ".route", "must name at least one HBM resource");
    std::set<std::string> route_set;
    for (const auto& key : result->route) {
        if (!route_set.insert(key).second) fail(path + ".route", "duplicate HBM resource");
    }
    result->label = string_value(wire.at("label"), path + ".label", true);
    result->service_factor = number_value(wire.at("service_factor"), path + ".service_factor");
    if (result->service_factor <= 0.0) fail(path + ".service_factor", "must be positive");
    const bool has_hardware = !wire.at("hardware_config").is_null();
    const bool has_stub = !wire.at("stub_phases").is_null();
    if (has_hardware && has_stub) fail(path, "production hardware and stub_phases are mutually exclusive");
    if (!has_hardware && !has_stub) fail(path + ".hardware_config", "production die plan requires resolved hardware");
    if (has_stub && !wire.at("specs").empty()) fail(path, "stub die plans must not include production specs");

    if (has_stub) {
        result->is_stub = true;
        array(wire.at("stub_phases"), path + ".stub_phases");
        for (size_t i = 0; i < wire.at("stub_phases").size(); ++i) {
            const auto& phase = wire.at("stub_phases")[i];
            const std::string pp = path + ".stub_phases[" + std::to_string(i) + "]";
            array(phase, pp);
            if (phase.empty() || !phase[0].is_string()) fail(pp, "phase must start with timer or memory");
            const auto kind = phase[0].get<std::string>();
            if (kind == "timer") {
                if (phase.size() != 2) fail(pp, "timer phase shape is [timer, ns]");
                result->stub_phases.push_back({NativeProjectionDiePhase::Kind::Timer,
                    uint_value(phase[1], pp + "[1]"), 0, 0, 0.0});
            } else if (kind == "memory") {
                if (phase.size() != 4) fail(pp, "memory phase shape is [memory, stage_id, actual_bytes, service_units]");
                result->stub_phases.push_back({NativeProjectionDiePhase::Kind::Memory,
                    0, uint_value(phase[1], pp + "[1]"), uint_value(phase[2], pp + "[2]"),
                    number_value(phase[3], pp + "[3]")});
            } else {
                fail(pp, "unknown stub phase kind");
            }
        }
        return result;
    }

    const auto& hardware_wire = wire.at("hardware_config");
    if (!hardware_wire.is_object()) fail(path + ".hardware_config", "must be object or null");
    const auto hardware = DieModel::make_hardware(hardware_wire);
    result->specs.reserve(wire.at("specs").size());
    result->plans.reserve(wire.at("specs").size());
    static std::mutex plan_cache_mutex;
    static DieModel::PlanCache plan_cache(4096);
    for (size_t i = 0; i < wire.at("specs").size(); ++i) {
        const auto spec = DieModel::parse_die_op_spec(wire.at("specs")[i]);
        const auto key = DieModel::plan_cache_key(*spec, hardware);
        std::shared_ptr<const DieModel::DiePlan> plan;
        {
            std::lock_guard<std::mutex> lock(plan_cache_mutex);
            plan = plan_cache.get(key);
            if (!plan) {
                plan = DieModel::compile(*spec, hardware);
                plan_cache.put(key, plan);
            }
        }
        result->specs.push_back(spec);
        result->plans.push_back(std::move(plan));
    }
    if (result->plans.empty()) fail(path + ".specs", "production die plan requires at least one spec");
    return result;
}

NativeProjectionLegSpec parse_leg_spec(const Json& wire,
                                       const std::string& path) {
    object(wire, path, {"leg_id", "kind", "route", "bytes_amount",
        "duration_ns", "base_ns", "group", "latency_ns", "ready_at_ns",
        "deps", "acquires", "releases", "signals_barrier",
        "remaining_bytes", "remaining_duration_ns", "in_service", "order",
        "die_plan"}, {"registration_ordinal"});
    NativeProjectionLegSpec result;
    result.leg_id = string_value(wire.at("leg_id"), path + ".leg_id");
    result.kind = string_value(wire.at("kind"), path + ".kind");
    result.route = parse_string_array(wire.at("route"), path + ".route");
    result.bytes_amount = uint_value(wire.at("bytes_amount"), path + ".bytes_amount");
    result.duration_ns = number_value(wire.at("duration_ns"), path + ".duration_ns");
    result.base_ns = number_value(wire.at("base_ns"), path + ".base_ns");
    result.group = string_value(wire.at("group"), path + ".group", true);
    result.latency_ns = uint_value(wire.at("latency_ns"), path + ".latency_ns");
    if (!wire.at("ready_at_ns").is_null()) {
        result.has_ready_at = true;
        result.ready_at_ns = uint_value(wire.at("ready_at_ns"), path + ".ready_at_ns");
    }
    result.deps = parse_string_array(wire.at("deps"), path + ".deps");
    result.acquires = parse_leases(wire.at("acquires"), path + ".acquires");
    result.releases = parse_leases(wire.at("releases"), path + ".releases");
    if (!wire.at("signals_barrier").is_null()) {
        result.has_signal_barrier = true;
        result.signals_barrier = string_value(wire.at("signals_barrier"), path + ".signals_barrier");
    }
    if (!wire.at("remaining_bytes").is_null()) {
        result.has_remaining_bytes = true;
        result.remaining_bytes = uint_value(wire.at("remaining_bytes"), path + ".remaining_bytes");
    }
    if (!wire.at("remaining_duration_ns").is_null()) {
        result.has_remaining_duration = true;
        result.remaining_duration_ns = number_value(
            wire.at("remaining_duration_ns"), path + ".remaining_duration_ns");
    }
    if (!wire.at("in_service").is_boolean()) fail(path + ".in_service", "expected boolean");
    result.in_service = wire.at("in_service").get<bool>();
    if (!wire.at("order").is_null()) {
        result.has_order = true;
        result.order = int64_value(wire.at("order"), path + ".order");
    }
    if (wire.contains("registration_ordinal") &&
        !wire.at("registration_ordinal").is_null()) {
        result.has_registration_ordinal = true;
        result.registration_ordinal = uint_value(
            wire.at("registration_ordinal"), path + ".registration_ordinal");
    }
    if (!wire.at("die_plan").is_null()) result.die_plan = parse_die_plan(wire.at("die_plan"), path + ".die_plan");
    return result;
}

NativeProjectionWorkSpec parse_work_spec(const Json& wire,
                                         const std::string& path,
                                         int expected_candidate) {
    object(wire, path, {"work_id", "legs", "ready_at_ns", "depends_on_work",
        "is_candidate", "service_leg_ids"}, {"pipeline_descriptors"});
    NativeProjectionWorkSpec result;
    if (wire.contains("pipeline_descriptors")) {
        array(wire.at("pipeline_descriptors"), path + ".pipeline_descriptors");
        result.pipeline_descriptors.reserve(wire.at("pipeline_descriptors").size());
        for (size_t i = 0; i < wire.at("pipeline_descriptors").size(); ++i) {
            try {
                result.pipeline_descriptors.push_back(PipelineDescriptor::parse(
                    wire.at("pipeline_descriptors")[i]));
            } catch (const std::exception& error) {
                fail(path + ".pipeline_descriptors[" + std::to_string(i) + "]",
                     error.what());
            }
        }
    }
    result.work_id = string_value(wire.at("work_id"), path + ".work_id");
    result.ready_at_ns = uint_value(wire.at("ready_at_ns"), path + ".ready_at_ns");
    if (!wire.at("is_candidate").is_boolean() ||
        (expected_candidate >= 0 &&
         wire.at("is_candidate").get<bool>() != (expected_candidate != 0))) {
        fail(path + ".is_candidate", "candidate flag mismatch");
    }
    result.is_candidate = wire.at("is_candidate").get<bool>();
    result.depends_on_work = parse_string_array(
        wire.at("depends_on_work"), path + ".depends_on_work");
    if (!wire.at("service_leg_ids").is_null()) {
        result.has_service_leg_ids = true;
        result.service_leg_ids = parse_string_array(
            wire.at("service_leg_ids"), path + ".service_leg_ids");
    }
    array(wire.at("legs"), path + ".legs");
    if (wire.at("legs").empty() && result.pipeline_descriptors.empty()) {
        fail(path + ".legs", "empty legs require pipeline_descriptors");
    }
    result.legs.reserve(wire.at("legs").size());
    for (size_t i = 0; i < wire.at("legs").size(); ++i) {
        result.legs.push_back(parse_leg_spec(
            wire.at("legs")[i], path + ".legs[" + std::to_string(i) + "]"));
    }
    return result;
}

std::shared_ptr<const NativeProjectionInput> parse_projection_input(
    const Json& wire) {
    validate_projection_input_shape(wire);
    auto result = std::make_shared<NativeProjectionInput>();
    result->now_ns = number_value(wire.at("now_ns"), "projection_input.now_ns");
    result->focus_work_id = string_value(
        wire.at("focus_work_id"), "projection_input.focus_work_id");
    result->resources.reserve(wire.at("resources").size());
    for (size_t i = 0; i < wire.at("resources").size(); ++i) {
        const auto& item = wire.at("resources")[i];
        result->resources.push_back({
            string_value(item.at("key"), "resource.key"),
            string_value(item.at("kind"), "resource.kind"),
            number_value(item.at("capacity_per_ns"), "resource.capacity_per_ns")});
    }
    result->background_works.reserve(wire.at("background_works").size());
    for (size_t i = 0; i < wire.at("background_works").size(); ++i) {
        result->background_works.push_back(parse_work_spec(
            wire.at("background_works")[i],
            "projection_input.background_works[" + std::to_string(i) + "]", -1));
        result->explicit_background_work_ids.push_back(
            result->background_works.back().work_id);
    }
    result->pipelines.reserve(wire.at("pipelines").size());
    for (size_t i = 0; i < wire.at("pipelines").size(); ++i) {
        const auto& item = wire.at("pipelines")[i];
        NativeProjectionBlockPipeline pipeline{
            string_value(item.at("pipeline_id"), "pipeline.pipeline_id"),
            string_value(item.at("fill_resource"), "pipeline.fill_resource"),
            uint_value(item.at("block_bytes"), "pipeline.block_bytes"),
            uint_value(item.at("num_blocks"), "pipeline.num_blocks"),
            number_value(item.at("consume_duration_ns"), "pipeline.consume_duration_ns"),
            string_value(item.at("buffer_gate_id"), "pipeline.buffer_gate_id"),
            uint_value(item.at("buffer_slots"), "pipeline.buffer_slots"),
            uint_value(item.at("startup_ns"), "pipeline.startup_ns")};
        result->pipelines.push_back(pipeline);
        NativeProjectionWorkSpec synthetic;
        synthetic.work_id = "pipeline:" + pipeline.pipeline_id;
        if (pipeline.num_blocks > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / 2)) {
            fail("pipeline.num_blocks", "expansion exceeds addressable task count");
        }
        synthetic.legs.reserve(static_cast<size_t>(pipeline.num_blocks * 2));
        for (uint64_t index = 0; index < pipeline.num_blocks; ++index) {
            NativeProjectionLegSpec fill;
            fill.leg_id = synthetic.work_id + "#fill" + std::to_string(index);
            fill.kind = "stream";
            fill.route.push_back(pipeline.fill_resource);
            fill.bytes_amount = pipeline.block_bytes;
            fill.latency_ns = index == 0 ? pipeline.startup_ns : 0;
            if (index != 0) {
                fill.deps.push_back(synthetic.work_id + "#fill" + std::to_string(index - 1));
            }
            fill.acquires.push_back({pipeline.buffer_gate_id, 1});
            synthetic.legs.push_back(std::move(fill));

            NativeProjectionLegSpec consume;
            consume.leg_id = synthetic.work_id + "#consume" + std::to_string(index);
            consume.kind = "fixed";
            consume.duration_ns = pipeline.consume_duration_ns;
            consume.deps.push_back(synthetic.work_id + "#fill" + std::to_string(index));
            if (index != 0) {
                consume.deps.push_back(synthetic.work_id + "#consume" + std::to_string(index - 1));
            }
            consume.releases.push_back({pipeline.buffer_gate_id, 1});
            synthetic.legs.push_back(std::move(consume));
        }
        result->background_works.push_back(std::move(synthetic));
    }
    result->gates.reserve(wire.at("gates").size());
    for (const auto& item : wire.at("gates")) {
        result->gates.push_back({
            string_value(item.at("gate_id"), "gate.gate_id"),
            string_value(item.at("kind"), "gate.kind"),
            uint_value(item.at("capacity_units"), "gate.capacity_units"),
            uint_value(item.at("initial_occupied"), "gate.initial_occupied")});
    }
    result->barriers.reserve(wire.at("barriers").size());
    for (const auto& item : wire.at("barriers")) {
        result->barriers.push_back({
            string_value(item.at("barrier_id"), "barrier.barrier_id"),
            parse_string_array(item.at("signallers"), "barrier.signallers")});
    }
    for (auto it = wire.at("efficiency").at("eta").begin();
         it != wire.at("efficiency").at("eta").end(); ++it) {
        result->eta.emplace(it.key(), number_value(it.value(), "efficiency.eta"));
    }
    for (auto it = wire.at("efficiency").at("gamma").begin();
         it != wire.at("efficiency").at("gamma").end(); ++it) {
        result->gamma.emplace(it.key(), number_value(it.value(), "efficiency.gamma"));
    }
    result->forecast_arrival_placeholders_ns.reserve(
        wire.at("forecast_arrival_placeholders_ns").size());
    for (const auto& tick : wire.at("forecast_arrival_placeholders_ns")) {
        result->forecast_arrival_placeholders_ns.push_back(
            uint_value(tick, "forecast arrival tick"));
    }
    if (wire.contains("forecast_start_index")) {
        const uint64_t index = uint_value(
            wire.at("forecast_start_index"), "forecast_start_index");
        if (index > result->background_works.size()) {
            fail("projection_input.forecast_start_index",
                 "exceeds the background work count");
        }
        result->forecast_start_index = static_cast<size_t>(index);
    }
    return result;
}

NativeProjectionOverlay parse_projection_overlay(const Json& wire) {
    object(wire, "private_overlay", {},
           {"resources", "extra_background_works", "gates", "barriers",
            "append_background_anchor"});
    NativeProjectionOverlay overlay;
    if (wire.contains("resources")) {
        overlay.resources.reserve(wire.at("resources").size());
        for (size_t i = 0; i < wire.at("resources").size(); ++i) {
            const auto& item = wire.at("resources")[i];
            const auto path =
                "private_overlay.resources[" + std::to_string(i) + "]";
            overlay.resources.push_back({
                string_value(item.at("key"), path + ".key"),
                string_value(item.at("kind"), path + ".kind"),
                number_value(item.at("capacity_per_ns"),
                             path + ".capacity_per_ns")});
        }
    }
    if (wire.contains("extra_background_works")) {
        overlay.extra_background_works.reserve(
            wire.at("extra_background_works").size());
        for (size_t i = 0; i < wire.at("extra_background_works").size(); ++i) {
            overlay.extra_background_works.push_back(parse_work_spec(
                wire.at("extra_background_works")[i],
                "private_overlay.extra_background_works["
                    + std::to_string(i) + "]", -1));
        }
    }
    if (wire.contains("gates")) {
        overlay.gates.reserve(wire.at("gates").size());
        for (size_t i = 0; i < wire.at("gates").size(); ++i) {
            const auto& item = wire.at("gates")[i];
            const auto path =
                "private_overlay.gates[" + std::to_string(i) + "]";
            overlay.gates.push_back({
                string_value(item.at("gate_id"), path + ".gate_id"),
                string_value(item.at("kind"), path + ".kind"),
                uint_value(item.at("capacity_units"), path + ".capacity_units"),
                uint_value(item.at("initial_occupied"),
                           path + ".initial_occupied")});
        }
    }
    if (wire.contains("barriers")) {
        overlay.barriers.reserve(wire.at("barriers").size());
        for (size_t i = 0; i < wire.at("barriers").size(); ++i) {
            const auto& item = wire.at("barriers")[i];
            const auto path =
                "private_overlay.barriers[" + std::to_string(i) + "]";
            overlay.barriers.push_back({
                string_value(item.at("barrier_id"), path + ".barrier_id"),
                parse_string_array(item.at("signallers"),
                                   path + ".signallers")});
        }
    }
    if (wire.contains("append_background_anchor")) {
        if (!wire.at("append_background_anchor").is_boolean()) {
            fail("private_overlay.append_background_anchor", "expected boolean");
        }
        overlay.append_background_anchor =
            wire.at("append_background_anchor").get<bool>();
    }
    return overlay;
}

/// Merge one private overlay onto the shared snapshot, reproducing the
/// legacy per-candidate snapshot field-for-field:
/// resources first-seen order with candidate additions appended (setdefault),
/// background order [shared] + [extras] + [forecast] via the splice index,
/// candidate barriers ahead of the shared (forecast) barriers, gates appended.
NativeProjectionInput apply_projection_overlay(
    const NativeProjectionInput& base, const NativeProjectionOverlay& overlay) {
    NativeProjectionInput merged;
    merged.now_ns = base.now_ns;
    merged.pipelines = base.pipelines;
    merged.eta = base.eta;
    merged.gamma = base.gamma;
    merged.forecast_arrival_placeholders_ns =
        base.forecast_arrival_placeholders_ns;
    merged.forecast_start_index = base.forecast_start_index;

    std::set<std::string> resource_keys;
    for (const auto& resource : base.resources) {
        resource_keys.insert(resource.key);
    }
    merged.resources = base.resources;
    for (const auto& resource : overlay.resources) {
        if (resource_keys.count(resource.key)) continue;  // setdefault
        resource_keys.insert(resource.key);
        merged.resources.push_back(resource);
    }

    std::set<std::string> work_ids;
    for (const auto& work : base.background_works) {
        work_ids.insert(work.work_id);
    }
    for (const auto& work : overlay.extra_background_works) {
        if (work_ids.count(work.work_id)) {
            fail("private_overlay.extra_background_works",
                 "duplicate background work id '" + work.work_id + "'");
        }
        work_ids.insert(work.work_id);
    }
    const size_t splice = merged.forecast_start_index ==
            static_cast<size_t>(-1)
        ? base.background_works.size() : merged.forecast_start_index;
    merged.background_works.reserve(
        base.background_works.size() + overlay.extra_background_works.size()
        + (overlay.append_background_anchor ? 1 : 0));
    for (size_t i = 0; i < splice; ++i) {
        merged.background_works.push_back(base.background_works[i]);
        merged.explicit_background_work_ids.push_back(
            base.background_works[i].work_id);
    }
    for (const auto& work : overlay.extra_background_works) {
        merged.background_works.push_back(work);
        merged.explicit_background_work_ids.push_back(work.work_id);
    }
    for (size_t i = splice; i < base.background_works.size(); ++i) {
        merged.background_works.push_back(base.background_works[i]);
        merged.explicit_background_work_ids.push_back(
            base.background_works[i].work_id);
    }
    if (overlay.append_background_anchor) {
        if (!merged.background_works.empty()) {
            fail("private_overlay.append_background_anchor",
                 "anchor requested with a non-empty background");
        }
        NativeProjectionWorkSpec anchor;
        anchor.work_id = "world:anchor";
        NativeProjectionLegSpec leg;
        leg.leg_id = "world:anchor#anchor";
        leg.kind = "fixed";
        leg.duration_ns = 0.0;
        anchor.legs.push_back(std::move(leg));
        merged.background_works.push_back(std::move(anchor));
        merged.explicit_background_work_ids.push_back("world:anchor");
    }
    merged.focus_work_id = merged.background_works.empty()
        ? base.focus_work_id : merged.background_works.front().work_id;

    merged.gates = base.gates;
    merged.gates.insert(merged.gates.end(),
                        overlay.gates.begin(), overlay.gates.end());
    // Legacy barrier order: candidate barriers first, forecast barriers after.
    merged.barriers = overlay.barriers;
    merged.barriers.insert(merged.barriers.end(),
                           base.barriers.begin(), base.barriers.end());
    return merged;
}

constexpr int kPending = 0;
constexpr int kTimer = 1;
constexpr int kGateWait = 2;
constexpr int kLatency = 3;
constexpr int kActive = 4;
constexpr int kDone = 5;
constexpr const char* kWorkNodePrefix = "work:";
constexpr const char* kStatusOk = "ok";
constexpr const char* kStatusWait = "wait_progress";
constexpr const char* kStatusBlocked = "blocked_no_progress";
constexpr const char* kStatusDegraded = "degraded";
constexpr const char* kStatusUnavailable = "model_unavailable";

double ulp(double value) {
    return std::nextafter(value, std::numeric_limits<double>::infinity()) - value;
}

uint64_t ceil_ns(double value) {
    if (value <= 0.0) return 0;
    const long double rounded = std::max(
        1.0L, std::ceil(static_cast<long double>(value)));
    if (!std::isfinite(value) ||
        rounded > static_cast<long double>(std::numeric_limits<uint64_t>::max())) {
        fail("completion", "ceil_ns overflows uint64");
    }
    return static_cast<uint64_t>(rounded);
}

uint64_t round_ties_even_u64(double value, const std::string& path) {
    if (!std::isfinite(value) || value < 0.0) fail(path, "must be finite and nonnegative");
    const double floor_value = std::floor(value);
    const double fraction = value - floor_value;
    double rounded = floor_value;
    if (fraction > 0.5 ||
        (fraction == 0.5 && std::fmod(floor_value, 2.0) != 0.0)) {
        rounded = floor_value + 1.0;
    }
    if (static_cast<long double>(rounded) >
        static_cast<long double>(std::numeric_limits<uint64_t>::max())) {
        fail(path, "rounded value overflows uint64");
    }
    return static_cast<uint64_t>(rounded);
}

struct NativeDieAction final {
    enum class Kind { Timer, Memory, Done } kind = Kind::Done;
    uint64_t ns = 0;
    uint64_t stage_id = 0;
    double actual_bytes = 0.0;
    double service_units = 0.0;
};

struct NativeDieTimerWindow final {
    PipelineWindowPlan plan;
    DieModel::InternalServiceWindowToken token;
    uint64_t original_service_ns = 0;
    uint64_t external_delay_ns = 0;
    bool uses_service_window = false;
    bool has_exact_feedback = false;
    std::string reason;
};

/// Per-world mutable DieModel execution chain. Plans are immutable and may
/// be shared through PlanCache; execution handles are always fresh per world.
class NativeDieExecution final {
  public:
    NativeDieExecution(
        std::shared_ptr<const NativeProjectionDiePlan> plan,
        const PipelineKernelContract* contract,
        const uint32_t effective_window_limit)
        : plan_(std::move(plan)), contract_(contract),
          effective_window_limit_(effective_window_limit) {
        if (!plan_) fail("die_plan", "missing parsed plan");
        if (effective_window_limit_ != 1 && effective_window_limit_ != 2 &&
            effective_window_limit_ != 4 && effective_window_limit_ != 8) {
            fail("die_plan", "effective window limit must be one of 1,2,4,8");
        }
        if (contract_ != nullptr) {
            if (effective_window_limit_ > contract_->window_tile_limit()) {
                fail("die_plan", "effective window limit exceeds run contract");
            }
            window_controller_ =
                std::make_unique<PipelineWindowController>(
                    contract_->window_tile_limit());
        }
        if (plan_->is_stub) return;
        executions_.reserve(plan_->plans.size());
        for (const auto& item : plan_->plans) {
            executions_.push_back(std::make_unique<DieModel::Execution>(item));
        }
    }

    NativeDieAction next_action() {
        if (closed_) return {NativeDieAction::Kind::Done, elapsed_ns_};
        if (plan_->is_stub) {
            if (stub_index_ >= plan_->stub_phases.size()) {
                return {NativeDieAction::Kind::Done, elapsed_ns_};
            }
            const auto& phase = plan_->stub_phases[stub_index_];
            if (phase.kind == NativeProjectionDiePhase::Kind::Timer) {
                return {NativeDieAction::Kind::Timer, phase.ns};
            }
            return {NativeDieAction::Kind::Memory, 0, phase.stage_id,
                    static_cast<double>(phase.actual_bytes),
                    phase.service_units * plan_->service_factor};
        }
        while (execution_index_ < executions_.size()) {
            const auto action = executions_[execution_index_]->next_action();
            if (action.kind == DieModel::ExecutionAction::Kind::Done) {
                if (execution_index_ + 1 == executions_.size()) {
                    return {NativeDieAction::Kind::Done,
                            add_elapsed(elapsed_ns_, action.ns)};
                }
                elapsed_ns_ = add_elapsed(elapsed_ns_, action.ns);
                ++execution_index_;
                continue;
            }
            if (action.kind == DieModel::ExecutionAction::Kind::Timer) {
                return {NativeDieAction::Kind::Timer, action.ns};
            }
            return {NativeDieAction::Kind::Memory, 0, action.stage_id,
                    action.actual_bytes,
                    action.service_units * plan_->service_factor};
        }
        return {NativeDieAction::Kind::Done, elapsed_ns_};
    }

    void advance_time(uint64_t ns) {
        if (closed_) return;
        if (plan_->is_stub) {
            if (stub_index_ >= plan_->stub_phases.size()) return;
            stub_clock_ = add_elapsed(stub_clock_, ns);
            const auto& phase = plan_->stub_phases[stub_index_];
            if (phase.kind == NativeProjectionDiePhase::Kind::Timer && ns >= phase.ns) {
                ++stub_index_;
                elapsed_ns_ = add_elapsed(elapsed_ns_, phase.ns);
            }
            return;
        }
        if (execution_index_ < executions_.size()) executions_[execution_index_]->advance_time(ns);
    }

    NativeDieTimerWindow prepare_timer_window(
        const uint32_t effective_limit) {
        NativeDieTimerWindow selected;
        if (plan_->is_stub) {
            selected.reason = "controlled_stub_phase_exact_only";
            last_reason_ = selected.reason;
            return selected;
        }
        if (contract_ == nullptr || window_controller_ == nullptr) {
            selected.reason = "missing_run_contract_exact_only";
            last_reason_ = selected.reason;
            return selected;
        }
        if (effective_limit <= 1) {
            selected.reason = "effective_window_limit_one";
            last_reason_ = selected.reason;
            return selected;
        }
        if (effective_limit > effective_window_limit_) {
            fail("die_plan", "window query loosens its frozen effective limit");
        }
        if (execution_index_ >= executions_.size()) {
            selected.reason = "no_current_die_execution";
            last_reason_ = selected.reason;
            return selected;
        }
        const auto info = executions_[execution_index_]->query_internal_timer_window(
            effective_limit);
        if (!info.supported || info.slices.empty()) {
            selected.reason = info.reason.empty()
                ? "typed_timer_window_unsupported" : info.reason;
            last_reason_ = selected.reason;
            return selected;
        }
        std::vector<uint64_t> original;
        std::vector<uint64_t> hardware_minimum;
        original.reserve(info.slices.size());
        hardware_minimum.reserve(info.slices.size());
        uint64_t span = 0;
        for (const auto& slice : info.slices) {
            if (slice.original_service_ns == 0 ||
                slice.original_service_ns >
                    std::numeric_limits<uint64_t>::max() - span) {
                fail("die_plan", "timer window service span is invalid");
            }
            span += slice.original_service_ns;
            original.push_back(slice.original_service_ns);
            hardware_minimum.push_back(slice.hardware_min_ns);
        }
        selected.plan = window_controller_->plan(
            *contract_, effective_limit, original, hardware_minimum, span);
        if (!selected.plan.accepted || selected.plan.selected_count == 0) {
            selected.reason = selected.plan.unsupported_reason.empty()
                ? "window_plan_rejected" : selected.plan.unsupported_reason;
            uncommitted_refinement_count_ = checked_quality_add(
                uncommitted_refinement_count_, selected.plan.refinement_count,
                "uncommitted refinement");
            last_reason_ = selected.reason;
            last_plan_ = selected.plan.to_json(*contract_, effective_limit);
            return selected;
        }
        if (selected.plan.selected_count > info.token.original_service_ticks.size()) {
            fail("die_plan", "window plan exceeds its immutable service token");
        }
        selected.original_service_ns = selected.plan.original_total_ticks;
        uint64_t assigned_total = 0;
        for (const auto assigned : selected.plan.assigned_service_ticks) {
            if (assigned > std::numeric_limits<uint64_t>::max() - assigned_total) {
                fail("die_plan", "assigned external window overflows uint64");
            }
            assigned_total += assigned;
        }
        selected.external_delay_ns = assigned_total;
        selected.has_exact_feedback = selected.plan.exact &&
            selected.plan.selected_count == 1;
        if (selected.plan.selected_count > 1) {
            selected.token = info.token;
            selected.token.original_service_ticks.resize(
                selected.plan.selected_count);
            selected.uses_service_window = true;
        } else if (!selected.has_exact_feedback) {
            selected.reason = "window_plan_not_committable";
            uncommitted_refinement_count_ = checked_quality_add(
                uncommitted_refinement_count_, selected.plan.refinement_count,
                "uncommitted refinement");
            last_reason_ = selected.reason;
            return selected;
        }
        if (selected.external_delay_ns != selected.original_service_ns) {
            fail("die_plan", "averaged timer window changed total external time");
        }
        last_reason_ = selected.reason;
        last_plan_ = selected.plan.to_json(*contract_, effective_limit);
        last_plan_["committed"] = false;
        return selected;
    }

    void advance_service_window(
        DieModel::InternalServiceWindowToken& token, uint64_t delta_ns) {
        if (plan_->is_stub || execution_index_ >= executions_.size()) {
            fail("die_execution", "window service has no real active DieModel execution");
        }
        executions_[execution_index_]->advance_internal_service_window(
            token, delta_ns);
    }

    void record_timer_callback(const uint64_t logical_actions,
                               const PipelineWindowPlan* plan = nullptr) {
        if (logical_actions == 0) {
            fail("die_execution", "timer callback must complete a logical action");
        }
        if (logical_actions > std::numeric_limits<uint64_t>::max() -
                                  logical_timer_event_count_ ||
            physical_timer_callback_count_ ==
                std::numeric_limits<uint64_t>::max()) {
            fail("die_execution", "timer callback counters overflow uint64");
        }
        logical_timer_event_count_ += logical_actions;
        ++physical_timer_callback_count_;
        if (logical_actions > 1) {
            if (coalesced_timer_callback_count_ ==
                std::numeric_limits<uint64_t>::max()) {
                fail("die_execution", "coalesced timer counter overflows uint64");
            }
            ++coalesced_timer_callback_count_;
        }
        if (plan != nullptr) {
            if (window_controller_ == nullptr || contract_ == nullptr) {
                fail("die_execution", "accepted timer window lost its controller");
            }
            window_controller_->record(*plan, true);
            last_plan_ = plan->to_json(*contract_, effective_window_limit_);
            last_plan_["committed"] = true;
            last_reason_.clear();
        }
    }

    void reject_timer_window(const NativeDieTimerWindow& selected,
                             const std::string& reason) {
        if (!selected.plan.accepted || selected.plan.selected_count <= 1) return;
        uncommitted_refinement_count_ = checked_quality_add(
            uncommitted_refinement_count_, selected.plan.refinement_count,
            "uncommitted refinement");
        last_reason_ = reason;
        last_plan_ = selected.plan.to_json(*contract_, effective_window_limit_);
        last_plan_["committed"] = false;
        last_plan_["external_boundary"] = reason;
    }

    nlohmann::json window_quality_json() const {
        const uint64_t windows = window_controller_ == nullptr
            ? 0 : window_controller_->window_count();
        const uint64_t refinements = checked_quality_add(
            window_controller_ == nullptr
                ? 0 : window_controller_->refinement_count(),
            uncommitted_refinement_count_, "refinement");
        const uint64_t prefix = window_controller_ == nullptr
            ? 0 : window_controller_->local_prefix_bound_max();
        nlohmann::json out{{"approximation_profile",
                            windows == 0 ? "exact"
                                         : "native_die_timer_window_v1"},
                           {"window_count", windows},
                           {"refinement_count", refinements},
                           {"local_prefix_bound_max", prefix},
                           {"logical_timer_event_count",
                            logical_timer_event_count_},
                           {"physical_timer_callback_count",
                            physical_timer_callback_count_},
                           {"coalesced_timer_callback_count",
                            coalesced_timer_callback_count_},
                           {"bound_kind", windows == 0
                                ? "none" : "local_prefix_service_ticks"},
                           {"fully_exact", windows == 0},
                           {"unsupported_reason", last_reason_.empty()
                                ? nlohmann::json(nullptr)
                                : nlohmann::json(last_reason_)},
                           {"last_window_plan", last_plan_}};
        if (contract_ != nullptr) {
            out["window_contract"] = {
                {"model_version", contract_->model_version()},
                {"timer_whitelist_version", contract_->timer_whitelist_version()},
                {"window_tile_limit", contract_->window_tile_limit()},
                {"effective_window_tile_limit", effective_window_limit_},
                {"window_local_error_ppm", contract_->window_local_error_ppm()},
                {"window_refine_margin_ppm", contract_->window_refine_margin_ppm()},
                {"prediction_config_sha256", contract_->prediction_config_sha256()}};
        } else {
            out["window_contract"] = nullptr;
        }
        return out;
    }

    void complete_memory(uint64_t stage_id) {
        if (closed_) fail("die_execution", "memory completion after close");
        if (plan_->is_stub) {
            if (stub_index_ >= plan_->stub_phases.size()) fail("die_execution", "no pending stub memory phase");
            const auto& phase = plan_->stub_phases[stub_index_];
            if (phase.kind != NativeProjectionDiePhase::Kind::Memory || phase.stage_id != stage_id) {
                fail("die_execution", "stub memory stage id does not match pending request");
            }
            ++stub_index_;
            elapsed_ns_ = stub_clock_;
            return;
        }
        if (execution_index_ >= executions_.size()) fail("die_execution", "no pending memory phase");
        executions_[execution_index_]->complete_memory(stage_id);
    }

    void close() {
        closed_ = true;
        executions_.clear();
    }

  private:
    std::shared_ptr<const NativeProjectionDiePlan> plan_;
    const PipelineKernelContract* contract_ = nullptr;
    uint32_t effective_window_limit_ = 1;
    std::unique_ptr<PipelineWindowController> window_controller_;
    uint64_t uncommitted_refinement_count_ = 0;
    uint64_t logical_timer_event_count_ = 0;
    uint64_t physical_timer_callback_count_ = 0;
    uint64_t coalesced_timer_callback_count_ = 0;
    std::string last_reason_;
    nlohmann::json last_plan_ = nullptr;
    std::vector<std::unique_ptr<DieModel::Execution>> executions_;
    size_t execution_index_ = 0;
    size_t stub_index_ = 0;
    uint64_t elapsed_ns_ = 0;
    uint64_t stub_clock_ = 0;
    bool closed_ = false;

    static uint64_t add_elapsed(uint64_t a, uint64_t b) {
        if (b > std::numeric_limits<uint64_t>::max() - a) {
            fail("die_execution", "elapsed time overflows uint64");
        }
        return a + b;
    }

    static uint64_t checked_quality_add(uint64_t a, uint64_t b,
                                        const char* label) {
        if (b > std::numeric_limits<uint64_t>::max() - a) {
            fail("die_execution", std::string(label) + " counter overflows uint64");
        }
        return a + b;
    }
};

struct RuntimeLeg final {
    const NativeProjectionLegSpec* spec = nullptr;
    // Compact pipeline legs own only their current materialized spec. Ordinary
    // work specs point into the immutable parsed input.
    std::unique_ptr<NativeProjectionLegSpec> owned_spec;
    std::string work_id;
    bool has_registration_ordinal = false;
    uint64_t registration_ordinal = 0;
    struct RuntimePipelineChain* pipeline_chain = nullptr;
    bool pipeline_fill = false;
    uint64_t pipeline_local_index = 0;
    int state = kPending;
    double remaining_bytes = 0.0;
    double remaining_duration = 0.0;
    bool has_completion = false;
    uint64_t completion = 0;
    bool timer_scheduled = false;
    bool in_active_list = false;
    std::unique_ptr<NativeDieExecution> die_execution;
    std::string die_phase_kind;
    double die_phase_ns = 0.0;
    uint64_t die_stage_id = 0;
    uint64_t die_actual_bytes_total = 0;
    double die_service_units_total = 0.0;
    uint64_t die_timer_logical_events = 1;
    bool die_timer_uses_window = false;
    bool die_timer_has_window_plan = false;
    PipelineWindowPlan die_timer_window_plan;
    DieModel::InternalServiceWindowToken die_timer_window_token;
};

struct RuntimePipelineChain final {
    const PipelineProjectionChain* spec = nullptr;
    std::shared_ptr<const PipelineDescriptor> descriptor;
    std::string work_id;
    uint64_t generated_count = 0;
    uint64_t completed_fill_count = 0;
    uint64_t completed_consume_count = 0;
    bool has_terminal_consume_completion = false;
    uint64_t terminal_consume_completion = 0;
    RuntimeLeg* initial_fill = nullptr;
    RuntimeLeg* initial_consume = nullptr;
};

struct VirtualPrefixBinding final {
    RuntimePipelineChain* chain = nullptr;
    bool is_fill = false;
};

struct VirtualIdRef final {
    RuntimePipelineChain* chain = nullptr;
    bool is_fill = false;
    uint64_t local_index = 0;
};

struct RuntimeGate final {
    const NativeProjectionGate* spec = nullptr;
    uint64_t occupied = 0;
    std::vector<RuntimeLeg*> waiters;
    uint64_t max_occupied = 0;
};

struct RuntimeBarrier final {
    const NativeProjectionBarrier* spec = nullptr;
    std::set<std::string> arrived;
    bool has_passed_at = false;
    uint64_t passed_at = 0;
};

struct ProjectionEvent final {
    double time_ns = 0.0;
    int priority = 0;
    std::string producer;
    uint64_t sequence = 0;
    int kind = 0;
    std::string task_id;
};

struct ProjectionEventLater final {
    bool operator()(const ProjectionEvent& a,
                    const ProjectionEvent& b) const {
        return std::tie(a.time_ns, a.priority, a.producer, a.sequence,
                        a.kind, a.task_id) >
               std::tie(b.time_ns, b.priority, b.producer, b.sequence,
                        b.kind, b.task_id);
    }
};

struct FinishTimes final {
    double boundary_contribution = 0.0;
    double completion = 0.0;
};

struct OrderProgression final {
    uint64_t first = 0;
    uint64_t stride = 1;
    uint64_t count = 0;
};

struct NativeBlockPipelineState final {
    double now_ns = 0.0;
    double rate_bns = 0.0;
    uint64_t blocks_remaining = 0;
    double in_fill_served_bytes = 0.0;
    uint64_t buffered_full_blocks = 0;
    double consuming_remaining_ns = 0.0;
};

struct NativeBlockPipelinePlan final {
    std::vector<double> fill_starts;
    std::vector<double> fill_ends;
    std::vector<double> consume_ends;
    std::vector<double> old_gate_release_times;
    double finish = 0.0;
};

__int128 projection_extended_gcd(__int128 a, __int128 b,
                                 __int128* x, __int128* y) {
    if (b == 0) { *x = 1; *y = 0; return a; }
    __int128 next_x = 0, next_y = 0;
    const __int128 gcd = projection_extended_gcd(b, a % b, &next_x, &next_y);
    *x = next_y;
    *y = next_x - (a / b) * next_y;
    return gcd;
}

bool order_progressions_intersect(const OrderProgression& a,
                                  const OrderProgression& b) {
    if (a.count == 0 || b.count == 0) return false;
    const unsigned __int128 a_last = static_cast<unsigned __int128>(a.first) +
        static_cast<unsigned __int128>(a.stride) * (a.count - 1);
    const unsigned __int128 b_last = static_cast<unsigned __int128>(b.first) +
        static_cast<unsigned __int128>(b.stride) * (b.count - 1);
    const unsigned __int128 low = std::max<unsigned __int128>(a.first, b.first);
    const unsigned __int128 high = std::min(a_last, b_last);
    if (low > high) return false;
    const uint64_t gcd = std::gcd(a.stride, b.stride);
    const __int128 delta = static_cast<__int128>(b.first) - a.first;
    if (delta % gcd != 0) return false;
    const uint64_t modulus = b.stride / gcd;
    uint64_t k = 0;
    if (modulus != 1) {
        __int128 inverse = 0, ignored = 0;
        projection_extended_gcd(a.stride / gcd, modulus, &inverse, &ignored);
        __int128 residue = ((delta / gcd) * inverse) % modulus;
        if (residue < 0) residue += modulus;
        k = static_cast<uint64_t>(residue);
    }
    const unsigned __int128 period =
        static_cast<unsigned __int128>(a.stride / gcd) * b.stride;
    unsigned __int128 first =
        static_cast<unsigned __int128>(a.first) +
        static_cast<unsigned __int128>(a.stride) * k;
    first %= period;
    if (first < low) {
        const unsigned __int128 distance = low - first;
        first += ((distance + period - 1) / period) * period;
    }
    return first <= high;
}

class NativeProjectionWorld final {
  public:
    explicit NativeProjectionWorld(
        const NativeProjectionInput& input,
        const PipelineKernelContract* contract = nullptr,
        const uint32_t effective_window_limit = 1,
        const uint64_t event_budget_limit = 0)
        : input_(input), now_ns_(input.now_ns), contract_(contract),
          effective_window_limit_(contract == nullptr
              ? 1U : effective_window_limit),
          event_budget_limit_(event_budget_limit) {
        if (effective_window_limit_ != 1 && effective_window_limit_ != 2 &&
            effective_window_limit_ != 4 && effective_window_limit_ != 8) {
            fail("window_limit", "effective limit must be one of 1,2,4,8");
        }
        if (contract_ != nullptr && effective_window_limit_ >
                contract_->window_tile_limit()) {
            fail("window_limit", "effective limit exceeds frozen run contract");
        }
        for (const auto& resource : input.resources) resources_.emplace(resource.key, &resource);
        for (const auto& gate : input.gates) {
            gates_.emplace(gate.gate_id, RuntimeGate{&gate, gate.initial_occupied,
                                                     {}, gate.initial_occupied});
        }
        for (const auto& barrier : input.barriers) {
            barriers_.emplace(barrier.barrier_id,
                              RuntimeBarrier{&barrier, {}, false, 0});
        }
        build();
    }

    std::string run(bool has_horizon, double horizon_ns,
                    uint64_t event_budget, bool drain_background,
                    bool analytic_segments) {
        event_budget_limit_ = event_budget;
        analytic_enabled_ = analytic_segments;
        while (true) {
            const auto focus_done = work_done_at_.find(input_.focus_work_id);
            const bool focus_complete = focus_done != work_done_at_.end() &&
                                        focus_done->second.has_value();
            if (focus_complete) {
                if (!drain_background) {
                    last_status_ = kStatusOk;
                    return last_status_;
                }
                bool unfinished = false;
                for (const auto& item : work_remaining_) {
                    if (item.second > 0) { unfinished = true; break; }
                }
                if (!unfinished) {
                    last_status_ = kStatusOk;
                    return last_status_;
                }
                const auto boundary = next_boundary();
                if (!boundary.has_value()) {
                    last_status_ = kStatusOk;
                    return last_status_;
                }
                if (event_budget != 0 && events_processed_ >= event_budget) {
                    last_status_ = kStatusOk;
                    return last_status_; // background drain is disclosure-only
                }
            } else {
                if (event_budget != 0 && events_processed_ >= event_budget) {
                    last_status_ = kStatusDegraded;
                    return last_status_;
                }
                const auto boundary = next_boundary();
                if (!boundary.has_value()) {
                    last_status_ = kStatusBlocked;
                    return last_status_;
                }
                if (has_horizon && *boundary > horizon_ns) {
                    last_status_ = kStatusWait;
                    return last_status_;
                }
            }
            if (!step(event_budget)) {
                const auto done = work_done_at_.find(input_.focus_work_id);
                if (done != work_done_at_.end() && done->second.has_value()) {
                    last_status_ = kStatusOk;
                    return last_status_;
                }
                last_status_ = kStatusBlocked;
                return last_status_;
            }
            if (last_step_budget_truncated_) {
                const auto done = work_done_at_.find(input_.focus_work_id);
                last_status_ = done != work_done_at_.end() && done->second.has_value()
                    ? kStatusOk : kStatusDegraded;
                return last_status_;
            }
        }
    }

    void set_conservation_check(bool enabled) { conservation_check_ = enabled; }

    uint64_t events_processed() const { return events_processed_; }
    uint64_t integration_steps() const { return integration_steps_; }
    double bytes_served_total() const { return bytes_served_total_; }
    size_t resource_count() const { return resources_.size(); }
    size_t work_count() const { return work_legs_.size(); }
    double now_ns() const { return now_ns_; }
    const std::string& last_status() const { return last_status_; }
    bool advance_one() { return step(); }

    Json window_quality_fields(const std::string& status) const {
        Json result{{"approximation_profile", "exact"},
                    {"window_count", 0}, {"refinement_count", 0},
                    {"local_prefix_bound_max", 0},
                    {"logical_event_count", events_processed_},
                    {"physical_step_count", integration_steps_},
                    {"bound_kind", "none"},
                    {"fully_exact", status != kStatusDegraded &&
                                         status != kStatusUnavailable},
                    {"unsupported_reason", nullptr}};
        if (contract_ == nullptr || effective_window_limit_ <= 1) return result;
        const auto add_counter = [](const uint64_t a, const uint64_t b,
                                    const char* const label) {
            if (b > std::numeric_limits<uint64_t>::max() - a) {
                fail("quality", std::string(label) + " overflows uint64");
            }
            return a + b;
        };
        uint64_t windows = 0;
        uint64_t refinements = 0;
        uint64_t prefix_bound = 0;
        uint64_t logical_timer_events = 0;
        uint64_t physical_timer_callbacks = 0;
        uint64_t coalesced_timer_callbacks = 0;
        std::string unsupported;
        bool has_die = false;
        for (const auto& entry : legs_) {
            const RuntimeLeg* leg = entry.second.get();
            if (!leg->die_execution) continue;
            has_die = true;
            const Json counters = leg->die_execution->window_quality_json();
            windows = add_counter(windows,
                counters.value("window_count", uint64_t{0}), "window count");
            refinements = add_counter(refinements,
                counters.value("refinement_count", uint64_t{0}),
                "refinement count");
            prefix_bound = std::max(prefix_bound,
                counters.value("local_prefix_bound_max", uint64_t{0}));
            logical_timer_events = add_counter(logical_timer_events,
                counters.value("logical_timer_event_count", uint64_t{0}),
                "DIE logical timer events");
            physical_timer_callbacks = add_counter(physical_timer_callbacks,
                counters.value("physical_timer_callback_count", uint64_t{0}),
                "DIE physical timer callbacks");
            coalesced_timer_callbacks = add_counter(coalesced_timer_callbacks,
                counters.value("coalesced_timer_callback_count", uint64_t{0}),
                "DIE coalesced callbacks");
            const Json reason = counters.value(
                "unsupported_reason", Json(nullptr));
            if (unsupported.empty() && !reason.is_null()) {
                unsupported = reason.get<std::string>();
            }
        }
        if (!has_die) return result;
        result["approximation_profile"] = windows == 0
            ? "exact" : "native_die_timer_window_v1";
        result["window_count"] = windows;
        result["refinement_count"] = refinements;
        result["local_prefix_bound_max"] = prefix_bound;
        result["bound_kind"] = windows == 0
            ? "none" : "local_prefix_service_ticks";
        result["fully_exact"] = windows == 0 &&
            status != kStatusDegraded && status != kStatusUnavailable;
        result["unsupported_reason"] = unsupported.empty()
            ? Json(nullptr) : Json(unsupported);
        result["die_logical_timer_event_count"] = logical_timer_events;
        result["die_physical_timer_callback_count"] = physical_timer_callbacks;
        result["die_coalesced_timer_callback_count"] = coalesced_timer_callbacks;
        result["window_contract"] = {
            {"model_version", contract_->model_version()},
            {"timer_whitelist_version", contract_->timer_whitelist_version()},
            {"window_tile_limit", contract_->window_tile_limit()},
            {"effective_window_tile_limit", effective_window_limit_},
            {"window_local_error_ppm", contract_->window_local_error_ppm()},
            {"window_refine_margin_ppm", contract_->window_refine_margin_ppm()},
            {"prediction_config_sha256", contract_->prediction_config_sha256()}};
        return result;
    }

    std::optional<uint64_t> work_completion(const std::string& id) const {
        const auto it = work_done_at_.find(id);
        return it == work_done_at_.end() ? std::nullopt : it->second;
    }

    const NativeProjectionInput& input() const { return input_; }

    std::optional<uint64_t> service_completion(const std::string& id) const {
        const NativeProjectionWorkSpec* work = nullptr;
        for (const auto& candidate : input_.background_works) {
            if (candidate.work_id == id) { work = &candidate; break; }
        }
        if (work == nullptr) return std::nullopt;
        std::vector<uint64_t> times;
        if (work->has_service_leg_ids) {
            for (const auto& leg_id : work->service_leg_ids) {
                const auto it = legs_.find(leg_id);
                if (it != legs_.end()) {
                    if (!it->second->has_completion) return std::nullopt;
                    times.push_back(it->second->completion);
                    continue;
                }
                const auto virtual_ref = find_virtual_ref(leg_id);
                if (!virtual_ref || virtual_ref->is_fill ||
                    virtual_ref->local_index + 1 != virtual_ref->chain->spec->tile_count) {
                    fail(id, "service_leg_ids contains an unknown or nonterminal virtual leg");
                }
                const auto* chain = virtual_ref->chain;
                if (!chain->has_terminal_consume_completion) return std::nullopt;
                times.push_back(chain->terminal_consume_completion);
            }
        } else {
            return work_completion(id);
        }
        if (times.empty()) return std::nullopt;
        return *std::max_element(times.begin(), times.end());
    }

    std::vector<std::string> attribution(const std::string& work_id) const {
        std::vector<std::string> lines;
        const auto found = work_legs_.find(work_id);
        if (found == work_legs_.end()) return lines;
        for (const auto* leg : found->second) {
            if (leg->state == kDone) continue;
            const auto& spec = *leg->spec;
            std::string line = spec.leg_id + ": ";
            if (leg->state == kPending) {
                line += "waiting deps [";
                auto deps = spec.deps;
                std::sort(deps.begin(), deps.end());
                for (size_t i = 0; i < deps.size(); ++i) {
                    if (i) line += ", ";
                    line += deps[i];
                }
                line += "]";
            } else if (leg->state == kTimer) {
                line += "ready_at ";
                line += spec.has_ready_at ? std::to_string(spec.ready_at_ns)
                                          : "work-level";
            } else if (leg->state == kGateWait) {
                line += "gate wait [";
                for (size_t i = 0; i < spec.acquires.size(); ++i) {
                    if (i) line += ", ";
                    const auto& lease = spec.acquires[i];
                    const auto& gate = gates_.at(lease.gate_id);
                    line += "'" + lease.gate_id + "(" + std::to_string(gate.occupied) +
                            "/" + std::to_string(gate.spec->capacity_units) + ")'";
                }
                line += "]";
            } else if (leg->state == kLatency) {
                line += "job latency " + std::to_string(spec.latency_ns) + "ns";
            } else if (leg->state == kActive) {
                line += "in service";
            }
            lines.push_back(std::move(line));
        }
        for (const auto& entry : legs_) {
            const auto* leg = entry.second.get();
            if (leg->pipeline_chain == nullptr || leg->work_id != work_id ||
                leg->state == kDone) continue;
            const auto& spec = *leg->spec;
            std::string line = spec.leg_id + ": ";
            if (leg->state == kPending) {
                line += "waiting deps [";
                auto deps = spec.deps;
                std::sort(deps.begin(), deps.end());
                for (size_t i = 0; i < deps.size(); ++i) {
                    if (i) line += ", ";
                    line += deps[i];
                }
                line += "]";
            } else if (leg->state == kGateWait) {
                line += "gate wait [";
                for (size_t i = 0; i < spec.acquires.size(); ++i) {
                    if (i) line += ", ";
                    const auto& lease = spec.acquires[i];
                    const auto& gate = gates_.at(lease.gate_id);
                    line += "'" + lease.gate_id + "(" + std::to_string(gate.occupied) +
                            "/" + std::to_string(gate.spec->capacity_units) + ")'";
                }
                line += "]";
            } else if (leg->state == kTimer) {
                line += "ready_at " + (spec.has_ready_at
                    ? std::to_string(spec.ready_at_ns) : std::string("work-level"));
            } else if (leg->state == kLatency) {
                line += "job latency " + std::to_string(spec.latency_ns) + "ns";
            } else if (leg->state == kActive) {
                line += "in service";
            }
            lines.push_back(std::move(line));
        }
        for (const auto& chain : pipeline_chains_) {
            if (chain->work_id == work_id && chain->generated_count < chain->spec->tile_count) {
                lines.push_back(chain->spec->fill.leg_id_prefix + "<cursor>: waiting for tile " +
                                std::to_string(chain->generated_count));
            }
        }
        return lines;
    }

    Json inspect() const {
        Json done = Json::object();
        for (const auto& entry : work_done_at_) {
            done[entry.first] = entry.second ? Json(*entry.second) : Json(nullptr);
        }
        Json leg_views = Json::object();
        for (const auto* leg : leg_order_) {
            const int64_t order = leg->spec->has_order ? leg->spec->order : 0;
            leg_views[leg->spec->leg_id] = {
                {"work_id", leg->work_id},
                {"state", state_name(leg->state)},
                {"remaining_bytes", leg->remaining_bytes},
                {"remaining_duration_ns", leg->remaining_duration},
                {"completion_ns", leg->has_completion ? Json(leg->completion) : Json(nullptr)},
                {"order_key", Json::array({order, leg->spec->leg_id})},
                {"die_actual_bytes_total", leg->die_actual_bytes_total},
                {"die_service_units_total", leg->die_service_units_total}};
        }
        Json pipeline_cursors = Json::array();
        for (const auto& chain : pipeline_chains_) {
            pipeline_cursors.push_back({
                {"work_id", chain->work_id},
                {"segment_id", chain->descriptor->identity().segment_id},
                {"group_ordinal", chain->spec->group_ordinal},
                {"rank", chain->spec->rank},
                {"tile_count", chain->spec->tile_count},
                {"generated_count", chain->generated_count},
                {"completed_fill_count", chain->completed_fill_count},
                {"completed_consume_count", chain->completed_consume_count},
                {"terminal_consume_completion_ns", chain->has_terminal_consume_completion
                    ? Json(chain->terminal_consume_completion) : Json(nullptr)}});
        }
        for (const auto& entry : legs_) {
            const auto* leg = entry.second.get();
            if (leg->pipeline_chain == nullptr) continue;
            const int64_t order = leg->spec->has_order ? leg->spec->order : 0;
            leg_views[leg->spec->leg_id] = {
                {"work_id", leg->work_id},
                {"state", state_name(leg->state)},
                {"remaining_bytes", leg->remaining_bytes},
                {"remaining_duration_ns", leg->remaining_duration},
                {"completion_ns", leg->has_completion ? Json(leg->completion) : Json(nullptr)},
                {"order_key", Json::array({order, leg->spec->leg_id})},
                {"die_actual_bytes_total", leg->die_actual_bytes_total},
                {"die_service_units_total", leg->die_service_units_total}};
        }
        Json gate_views = Json::object();
        for (const auto& entry : gates_) {
            Json waiters = Json::array();
            for (const auto* waiter : entry.second.waiters) {
                waiters.push_back(waiter->spec->leg_id);
            }
            gate_views[entry.first] = {
                {"kind", entry.second.spec->kind},
                {"capacity", entry.second.spec->capacity_units},
                {"initial_occupied", entry.second.spec->initial_occupied},
                {"occupied", entry.second.occupied},
                {"max_occupied", entry.second.max_occupied},
                {"waiting_leg_ids", std::move(waiters)}};
        }
        Json barrier_views = Json::object();
        for (const auto& entry : barriers_) {
            Json arrived = Json::array();
            for (const auto& leg_id : entry.second.arrived) arrived.push_back(leg_id);
            Json signallers = entry.second.spec->signallers;
            barrier_views[entry.first] = {
                {"signallers", std::move(signallers)},
                {"arrived", std::move(arrived)},
                {"has_passed", entry.second.has_passed_at},
                {"passed_at_ns", entry.second.has_passed_at
                    ? Json(entry.second.passed_at) : Json(nullptr)}};
        }
        return Json{{"now_ns", now_ns_}, {"status", last_status_},
                    {"events_processed", events_processed_},
                    {"integration_steps", integration_steps_},
                    {"work_done_at", std::move(done)},
                    {"work_remaining", work_remaining_},
                    {"legs", std::move(leg_views)},
                    {"pipeline_cursors", std::move(pipeline_cursors)},
                    {"gates", std::move(gate_views)},
                    {"barriers", std::move(barrier_views)},
                    {"cost_disclosure", cost_disclosure()}};
    }

    void set_clock(double time_ns) {
        if (!std::isfinite(time_ns) || time_ns < 0.0) {
            fail("set_clock", "time must be finite and nonnegative");
        }
        now_ns_ = time_ns;
    }

    Json integrate_controlled(double target_ns) {
        if (!std::isfinite(target_ns) || target_ns < now_ns_) {
            fail("integrate", "target must be finite and >= current time");
        }
        double pre_remaining = 0.0;
        Json rates = Json::array();
        for (const auto* leg : active_legs_) {
            if ((leg->spec->kind == "stream" || leg->spec->kind == "compute") &&
                leg->remaining_bytes > 0.0) {
                const double rate = rate_of(leg);
                pre_remaining += leg->remaining_bytes;
                rates.push_back(Json::array({leg->spec->leg_id, rate}));
            }
        }
        const double before = now_ns_;
        const double bytes_before = bytes_served_total_;
        integrate(target_ns);
        const double served_total = bytes_served_total_ - bytes_before;
        return Json{{"pre_remaining", pre_remaining},
                    {"served_total", served_total},
                    {"rates", std::move(rates)},
                    {"interval_start_ns", before}, {"now_ns", now_ns_}};
    }

    void assert_conservation_controlled(double pre_remaining,
                                        double served_total,
                                        const Json& rates_wire,
                                        double interval_start_ns,
                                        double reported_now_ns) const {
        array(rates_wire, "assert_conservation.rates");
        std::vector<std::pair<std::string, double>> rates;
        for (size_t i = 0; i < rates_wire.size(); ++i) {
            const auto& item = rates_wire[i];
            if (!item.is_array() || item.size() != 2) {
                fail("assert_conservation.rates", "expected ordered [leg_id, rate] pairs");
            }
            rates.emplace_back(string_value(item[0], "assert_conservation.leg_id"),
                               number_value(item[1], "assert_conservation.rate"));
        }
        double post_remaining = 0.0;
        for (const auto* leg : active_legs_) {
            if (leg->remaining_bytes > 0.0) post_remaining += leg->remaining_bytes;
        }
        double rate_sum = 0.0;
        double expected = 0.0;
        for (const auto& rate : rates) rate_sum += rate.second;
        const double slack = rate_sum * ulp(std::max(1.0, reported_now_ns));
        if (std::abs(pre_remaining - post_remaining - served_total) >
            1e-6 * std::max(1.0, pre_remaining) + slack) {
            fail("conservation", "byte conservation violated: remaining decrease differs from served bytes");
        }
        const double elapsed = reported_now_ns - interval_start_ns;
        for (const auto& rate : rates) expected += rate.second * elapsed;
        if (std::abs(served_total - expected) >
            1e-6 * std::max(1.0, std::abs(expected)) + slack) {
            fail("conservation", "time-service linkage violated: served bytes differ from rate*time");
        }
    }

  private:
    const NativeProjectionInput& input_;
    double now_ns_;
    const PipelineKernelContract* contract_ = nullptr;
    uint32_t effective_window_limit_ = 1;
    uint64_t event_budget_limit_ = 0;
    std::map<std::string, const NativeProjectionResource*> resources_;
    std::map<std::string, RuntimeGate> gates_;
    std::map<std::string, RuntimeBarrier> barriers_;
    std::map<std::string, std::unique_ptr<RuntimeLeg>> legs_;
    std::vector<std::unique_ptr<RuntimePipelineChain>> pipeline_chains_;
    std::map<std::string, std::vector<VirtualPrefixBinding>> virtual_prefixes_;
    std::vector<RuntimeLeg*> active_legs_; // exact activation/insertion order
    std::vector<RuntimeLeg*> leg_order_;   // original work/leg registration order
    std::map<std::string, std::set<std::string>> active_by_resource_;
    std::map<std::string, std::vector<RuntimeLeg*>> work_legs_;
    std::map<std::string, uint64_t> work_logical_leg_count_;
    std::map<std::string, uint64_t> work_completed_leg_count_;
    std::map<std::string, uint64_t> work_max_completion_;
    std::map<std::string, uint64_t> work_remaining_;
    std::map<std::string, std::optional<uint64_t>> work_done_at_;
    std::map<std::string, const NativeProjectionWorkSpec*> work_specs_;
    std::map<std::string, std::vector<std::string>> dependents_;
    std::map<std::string, uint64_t> deps_remaining_;
    std::priority_queue<ProjectionEvent, std::vector<ProjectionEvent>,
                        ProjectionEventLater> events_;
    uint64_t sequence_ = 0;
    uint64_t events_processed_ = 0;
    uint64_t integration_steps_ = 0;
    double bytes_served_total_ = 0.0;
    std::string last_status_ = kStatusWait;
    bool conservation_check_ = true;
    bool last_step_budget_truncated_ = false;
    bool dependencies_wired_ = false;
    bool analytic_enabled_ = false;

    uint32_t remaining_window_limit() const {
        if (contract_ == nullptr || effective_window_limit_ <= 1) return 1;
        if (event_budget_limit_ == 0) return effective_window_limit_;
        if (events_processed_ >= event_budget_limit_) return 1;
        const uint64_t remaining = event_budget_limit_ - events_processed_;
        uint32_t budget_limit = 1;
        for (const uint32_t candidate : {2U, 4U, 8U}) {
            if (candidate <= remaining) budget_limit = candidate;
        }
        return std::min(effective_window_limit_, budget_limit);
    }

    bool has_external_boundary_through(const RuntimeLeg* current,
                                       const double end_time_ns) const {
        const auto current_finishes = finishes();
        for (const auto& entry : current_finishes) {
            if (entry.first != current->spec->leg_id &&
                entry.second.boundary_contribution <= end_time_ns) {
                return true;
            }
        }
        // Any due event can change readiness, resource membership, or the
        // event-budget prefix before this timer window settles. Stale events
        // only cause a conservative exact fallback here.
        return !events_.empty() && events_.top().time_ns <= end_time_ns;
    }

    static const char* state_name(int state) {
        switch (state) {
            case kPending: return "pending";
            case kTimer: return "timer";
            case kGateWait: return "gate_wait";
            case kLatency: return "latency";
            case kActive: return "active";
            case kDone: return "done";
            default: return "unknown";
        }
    }

    Json cost_disclosure() const {
        Json out{{"events_processed", events_processed_},
                    {"integration_steps", integration_steps_},
                    {"resource_count", resources_.size()},
                    {"work_count", work_legs_.size()},
                    {"leg_count", logical_leg_count()},
                    {"gate_count", gates_.size()},
                    {"barrier_count", barriers_.size()},
                    {"bytes_served_total", static_cast<uint64_t>(bytes_served_total_)}};
        uint64_t die_count = 0, actual_bytes = 0;
        double service_units = 0.0;
        for (const auto* leg : leg_order_) {
            if (!leg->spec->die_plan) continue;
            ++die_count;
            actual_bytes += leg->die_actual_bytes_total;
            service_units += leg->die_service_units_total;
        }
        if (die_count != 0) {
            out["die_plan_leg_count"] = die_count;
            out["die_actual_bytes_total"] = actual_bytes;
            out["die_service_units_total"] = static_cast<uint64_t>(service_units);
        }
        return out;
    }

    uint64_t logical_leg_count() const {
        uint64_t total = 0;
        for (const auto& entry : work_logical_leg_count_) {
            if (entry.second > std::numeric_limits<uint64_t>::max() - total) {
                fail("cost_disclosure.leg_count", "sum overflows uint64");
            }
            total += entry.second;
        }
        return total;
    }

    double eta_of(const std::string& key) const {
        auto it = input_.eta.find(key);
        if (it != input_.eta.end()) return it->second;
        const auto colon = key.find(':');
        const std::string prefix = key.substr(0, colon);
        it = input_.eta.find(prefix);
        return it != input_.eta.end() ? it->second : 1.0;
    }

    double gamma_of(const std::string& group) const {
        const auto it = input_.gamma.find(group);
        return it == input_.gamma.end() ? 1.0 : it->second;
    }

    void build() {
        std::map<std::string, std::string> barrier_for_signaller;
        for (const auto& entry : barriers_) {
            for (const auto& signaller : entry.second.spec->signallers) {
                if (!barrier_for_signaller.emplace(signaller, entry.first).second) {
                    fail("barriers", "a leg cannot signal two barriers");
                }
            }
        }
        for (const auto& work : input_.background_works) register_work(work, barrier_for_signaller);
        for (const auto& barrier : barriers_) deps_remaining_.emplace(barrier.first, 0);
        wire_dependencies();
        check_acyclic();
        for (const auto& work : input_.background_works) {
            std::vector<RuntimeLeg*> initial = work_legs_.at(work.work_id);
            if (!work.pipeline_descriptors.empty()) {
                for (const auto& chain : pipeline_chains_) {
                    if (chain->work_id != work.work_id) continue;
                    if (chain->initial_fill) initial.push_back(chain->initial_fill);
                    if (chain->initial_consume) initial.push_back(chain->initial_consume);
                }
                std::sort(initial.begin(), initial.end(), [](const auto* a, const auto* b) {
                    if (!a->has_registration_ordinal || !b->has_registration_ordinal) {
                        fail("pipeline registration", "missing registration_ordinal");
                    }
                    return std::tie(a->registration_ordinal, a->spec->leg_id) <
                           std::tie(b->registration_ordinal, b->spec->leg_id);
                });
            }
            for (auto* leg : initial) {
                if (leg->spec->in_service) enter_service(leg);
                else if (deps_remaining_.at(leg->spec->leg_id) == 0) on_deps_satisfied(leg);
            }
        }
        for (const auto tick : input_.forecast_arrival_placeholders_ns) {
            push(static_cast<double>(tick), 3, "forecast", 3, "");
        }
    }

    std::string pipeline_leg_id(const RuntimePipelineChain& chain,
                                bool is_fill, uint64_t local_index) const {
        if (is_fill) {
            return chain.spec->fill.leg_id_prefix +
                std::to_string(chain.spec->fill.first_index + local_index);
        }
        return chain.spec->consume.leg_id_prefix +
            std::to_string(chain.spec->consume.first_index + local_index);
    }

    std::optional<VirtualIdRef> find_virtual_ref(const std::string& id) const {
        std::optional<VirtualIdRef> result;
        for (const auto& prefix_entry : virtual_prefixes_) {
            const auto& prefix = prefix_entry.first;
            if (id.size() <= prefix.size() || id.compare(0, prefix.size(), prefix) != 0) continue;
            const auto suffix = id.substr(prefix.size());
            uint64_t index = 0;
            for (const auto ch : suffix) {
                if (ch < '0' || ch > '9') { index = std::numeric_limits<uint64_t>::max(); break; }
                const uint64_t digit = static_cast<uint64_t>(ch - '0');
                if (index > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
                    index = std::numeric_limits<uint64_t>::max();
                    break;
                }
                index = index * 10 + digit;
            }
            if (index == std::numeric_limits<uint64_t>::max() ||
                std::to_string(index) != suffix) continue;
            for (const auto& binding : prefix_entry.second) {
                const uint64_t first_index = binding.is_fill
                    ? binding.chain->spec->fill.first_index
                    : binding.chain->spec->consume.first_index;
                if (index < first_index ||
                    index - first_index >= binding.chain->spec->tile_count) continue;
                if (result.has_value()) fail(id, "ambiguous compact virtual leg ID");
                result = VirtualIdRef{binding.chain, binding.is_fill,
                                      index - first_index};
            }
        }
        return result;
    }

    bool node_is_completed(const std::string& id) const {
        if (has_prefix(id, kWorkNodePrefix)) {
            const auto work_id = id.substr(std::char_traits<char>::length(kWorkNodePrefix));
            const auto it = work_done_at_.find(work_id);
            return it != work_done_at_.end() && it->second.has_value();
        }
        const auto barrier = barriers_.find(id);
        if (barrier != barriers_.end()) return barrier->second.has_passed_at;
        const auto leg = legs_.find(id);
        if (leg != legs_.end()) {
            if (leg->second->pipeline_chain != nullptr) {
                const auto* chain = leg->second->pipeline_chain;
                return leg->second->pipeline_fill
                    ? leg->second->pipeline_local_index < chain->completed_fill_count
                    : leg->second->pipeline_local_index < chain->completed_consume_count;
            }
            return leg->second->state == kDone && leg->second->has_completion;
        }
        const auto virtual_ref = find_virtual_ref(id);
        if (!virtual_ref) return false;
        return virtual_ref->is_fill
            ? virtual_ref->local_index < virtual_ref->chain->completed_fill_count
            : virtual_ref->local_index < virtual_ref->chain->completed_consume_count;
    }

    void register_leg_dependencies(RuntimeLeg* leg) {
        const auto work_it = work_specs_.find(leg->work_id);
        if (work_it == work_specs_.end()) fail(leg->work_id, "missing registered work spec");
        const auto& work = *work_it->second;
        std::vector<std::string> deps = leg->spec->deps;
        if (deps.empty() && !leg->spec->has_ready_at) {
            for (const auto& dep_work : work.depends_on_work) {
                deps.push_back(std::string(kWorkNodePrefix) + dep_work);
            }
        }
        uint64_t count = 0;
        for (const auto& dep : deps) {
            const bool work_node = has_prefix(dep, kWorkNodePrefix);
            const bool known_work_node = work_node &&
                work_legs_.count(dep.substr(std::char_traits<char>::length(kWorkNodePrefix)));
            const bool known = legs_.count(dep) || barriers_.count(dep) ||
                               find_virtual_ref(dep).has_value() || known_work_node;
            if (!known) fail(leg->spec->leg_id, "unknown dependency " + dep);
            if (work_node && !known_work_node) {
                fail(leg->spec->leg_id, "unknown work dependency " + dep);
            }
            if (node_is_completed(dep)) continue;
            auto& targets = dependents_[dep];
            targets.push_back(leg->spec->leg_id);
            std::sort(targets.begin(), targets.end());
            ++count;
        }
        deps_remaining_[leg->spec->leg_id] = count;
    }

    uint64_t projection_share_at(const PipelineProjectionShare& share,
                                 uint64_t local_index) const {
        const uint64_t global_index = share.share_index_first + local_index;
        const uint64_t quotient = share.total_units / share.total_tiles;
        const uint64_t remainder = share.total_units % share.total_tiles;
        return quotient + (global_index < remainder ? 1 : 0);
    }

    RuntimeLeg* create_pipeline_leg(RuntimePipelineChain* chain,
                                    bool is_fill, uint64_t local_index) {
        const auto& projection = *chain->spec;
        if (local_index >= projection.tile_count) fail(chain->work_id, "pipeline tile index outside chain");
        auto owned_spec = std::make_unique<NativeProjectionLegSpec>();
        auto& spec = *owned_spec;
        spec.leg_id = pipeline_leg_id(*chain, is_fill, local_index);
        spec.kind = is_fill ? "stream" : projection.consume.kind;
        spec.route = is_fill ? projection.fill.route : projection.consume.route;
        spec.group = is_fill ? std::string() : projection.consume.group;
        spec.latency_ns = is_fill ? projection.fill.latency_ns :
            (local_index == 0 ? projection.consume.latency_first_ns
                              : projection.consume.latency_rest_ns);
        if (is_fill) {
            const auto group = std::find_if(
                chain->descriptor->groups().begin(),
                chain->descriptor->groups().end(),
                [&](const auto& candidate) {
                    return candidate.group_ordinal == projection.group_ordinal;
                });
            if (group == chain->descriptor->groups().end()) {
                fail(chain->work_id, "pipeline fill references missing group");
            }
            const auto& axis = group->tile_axis;
            const unsigned __int128 consumed_units =
                static_cast<unsigned __int128>(axis.units_per_tile) * local_index;
            if (consumed_units >= axis.total_units) {
                fail(chain->work_id, "pipeline fill tile begins beyond group extent");
            }
            const uint64_t remaining_units = axis.total_units -
                static_cast<uint64_t>(consumed_units);
            spec.bytes_amount = std::min(projection.fill.bytes_per_tile,
                                         remaining_units);
            spec.acquires.push_back({projection.fill.gate_id, 1});
            if (local_index == 0) spec.deps = projection.fill.start_deps;
            else spec.deps.push_back(pipeline_leg_id(*chain, true, local_index - 1));
        } else {
            if (projection.consume.kind == "compute") {
                if (!projection.consume.has_compute_share || !projection.consume.has_hbm_share) {
                    fail(chain->work_id, "compute chain is missing byte-share vectors");
                }
                spec.bytes_amount = static_cast<uint64_t>(projection_share_at(
                    projection.consume.hbm_share, local_index));
                spec.base_ns = static_cast<double>(projection_share_at(
                    projection.consume.compute_share, local_index));
            }
            spec.deps = projection.consume.consume_deps;
            spec.deps.push_back(pipeline_leg_id(*chain, true, local_index));
            if (local_index == 0) {
                spec.deps.insert(spec.deps.end(),
                    projection.consume.consume_start_deps.begin(),
                    projection.consume.consume_start_deps.end());
            } else {
                spec.deps.push_back(pipeline_leg_id(*chain, false, local_index - 1));
            }
            spec.releases.push_back({projection.consume.gate_id, 1});
        }
        const bool has_completion_order = is_fill
            ? projection.fill.has_completion_order_first
            : projection.consume.has_completion_order_first;
        const int64_t completion_order_first = is_fill
            ? projection.fill.completion_order_first
            : projection.consume.completion_order_first;
        const int64_t completion_order_stride = is_fill
            ? projection.fill.completion_order_stride
            : projection.consume.completion_order_stride;
        spec.has_order = has_completion_order;
        if (has_completion_order) {
            const __int128 order = static_cast<__int128>(completion_order_first) +
                static_cast<__int128>(completion_order_stride) * local_index;
            if (order < std::numeric_limits<int64_t>::min() ||
                order > std::numeric_limits<int64_t>::max()) {
                fail(spec.leg_id, "completion-order value overflows int64");
            }
            spec.order = static_cast<int64_t>(order);
        }
        spec.has_registration_ordinal = true;
        spec.registration_ordinal = is_fill
            ? projection.fill.fill_order_first + local_index * projection.fill.fill_order_stride
            : projection.consume.consume_order_first + local_index * projection.consume.consume_order_stride;

        if (legs_.count(spec.leg_id)) fail(spec.leg_id, "duplicate pipeline logical leg ID");
        auto leg = std::make_unique<RuntimeLeg>();
        leg->owned_spec = std::move(owned_spec);
        leg->spec = leg->owned_spec.get();
        leg->work_id = chain->work_id;
        leg->has_registration_ordinal = true;
        leg->registration_ordinal = spec.registration_ordinal;
        leg->pipeline_chain = chain;
        leg->pipeline_fill = is_fill;
        leg->pipeline_local_index = local_index;
        leg->remaining_bytes = static_cast<double>(spec.bytes_amount);
        if (spec.kind == "fixed") leg->remaining_duration = spec.duration_ns;
        else if (spec.kind == "compute") leg->remaining_duration = spec.base_ns * gamma_of(spec.group);
        auto* raw = leg.get();
        legs_.emplace(spec.leg_id, std::move(leg));
        deps_remaining_.emplace(spec.leg_id, 0);
        if (dependencies_wired_) register_leg_dependencies(raw);
        return raw;
    }

    void materialize_pipeline_pair(RuntimePipelineChain* chain,
                                   uint64_t local_index) {
        if (local_index != chain->generated_count ||
            local_index >= chain->spec->tile_count) {
            fail(chain->work_id, "pipeline cursor generated out of sequence");
        }
        auto* fill = create_pipeline_leg(chain, true, local_index);
        auto* consume = create_pipeline_leg(chain, false, local_index);
        if (local_index == 0) {
            chain->initial_fill = fill;
            chain->initial_consume = consume;
        }
        ++chain->generated_count;
    }

    void register_work(const NativeProjectionWorkSpec& work,
                       const std::map<std::string, std::string>& barrier_for_signaller) {
        if (work_legs_.count(work.work_id)) fail("works", "duplicate work id");
        std::vector<RuntimeLeg*> stored;
        stored.reserve(work.legs.size());
        for (const auto& spec : work.legs) {
            if (legs_.count(spec.leg_id)) fail("works", "duplicate leg id");
            const auto expected_barrier = barrier_for_signaller.find(spec.leg_id);
            const bool declared = expected_barrier != barrier_for_signaller.end();
            if (declared != spec.has_signal_barrier ||
                (declared && expected_barrier->second != spec.signals_barrier)) {
                fail("works", "leg barrier signal disagrees with barrier registry");
            }
            for (const auto& key : spec.route) {
                if (!resources_.count(key)) fail(spec.leg_id, "unknown route resource " + key);
            }
            for (const auto& lease : spec.acquires) {
                if (!gates_.count(lease.gate_id)) fail(spec.leg_id, "unknown acquire gate " + lease.gate_id);
            }
            for (const auto& lease : spec.releases) {
                if (!gates_.count(lease.gate_id)) fail(spec.leg_id, "unknown release gate " + lease.gate_id);
            }
            auto leg = std::make_unique<RuntimeLeg>();
            leg->spec = &spec;
            leg->work_id = work.work_id;
            leg->has_registration_ordinal = spec.has_registration_ordinal;
            leg->registration_ordinal = spec.registration_ordinal;
            leg->remaining_bytes = static_cast<double>(
                spec.has_remaining_bytes ? spec.remaining_bytes : spec.bytes_amount);
            if (spec.has_remaining_duration) leg->remaining_duration = spec.remaining_duration_ns;
            else if (spec.kind == "fixed") leg->remaining_duration = spec.duration_ns;
            else if (spec.kind == "compute") leg->remaining_duration = spec.base_ns * gamma_of(spec.group);
            auto* raw = leg.get();
            legs_.emplace(spec.leg_id, std::move(leg));
            leg_order_.push_back(raw);
            stored.push_back(raw);
            deps_remaining_.emplace(spec.leg_id, 0);
        }

        uint64_t logical_count = static_cast<uint64_t>(work.legs.size());
        for (const auto& descriptor : work.pipeline_descriptors) {
            if (!descriptor) fail(work.work_id, "null pipeline descriptor");
            for (const auto& projection : descriptor->projection().chains) {
                for (const auto& route : projection.fill.route) {
                    if (!resources_.count(route)) fail(work.work_id, "unknown pipeline fill resource " + route);
                }
                for (const auto& route : projection.consume.route) {
                    if (!resources_.count(route)) fail(work.work_id, "unknown pipeline consume resource " + route);
                }
                const auto gate = gates_.find(projection.fill.gate_id);
                if (gate == gates_.end()) fail(work.work_id, "unknown projection gate " + projection.fill.gate_id);
                if (gate->second.spec->capacity_units != projection.fill.gate_capacity) {
                    fail(work.work_id, "projection gate capacity disagrees with input gate");
                }
                if (projection.fill.gate_id != projection.consume.gate_id) {
                    fail(work.work_id, "fill and consume gate IDs disagree");
                }
                if (projection.tile_count > std::numeric_limits<uint64_t>::max() / 2 ||
                    logical_count > std::numeric_limits<uint64_t>::max() - projection.tile_count * 2) {
                    fail(work.work_id, "pipeline logical leg count overflows uint64");
                }
                logical_count += projection.tile_count * 2;
                auto chain = std::make_unique<RuntimePipelineChain>();
                chain->spec = &projection;
                chain->descriptor = descriptor;
                chain->work_id = work.work_id;
                auto* raw_chain = chain.get();
                pipeline_chains_.push_back(std::move(chain));
                virtual_prefixes_[projection.fill.leg_id_prefix].push_back({raw_chain, true});
                virtual_prefixes_[projection.consume.leg_id_prefix].push_back({raw_chain, false});
            }
        }
        std::vector<OrderProgression> work_registration_orders;
        std::vector<std::pair<std::string, std::pair<uint64_t, uint64_t>>> work_id_ranges;
        for (const auto& chain : pipeline_chains_) {
            if (chain->work_id != work.work_id) continue;
            const auto add_range = [&](const PipelineProjectionFill& item) {
                work_id_ranges.push_back({item.leg_id_prefix,
                    {item.first_index, item.first_index + chain->spec->tile_count}});
            };
            add_range(chain->spec->fill);
            work_id_ranges.push_back({chain->spec->consume.leg_id_prefix,
                {chain->spec->consume.first_index,
                 chain->spec->consume.first_index + chain->spec->tile_count}});
            work_registration_orders.push_back({chain->spec->fill.fill_order_first,
                chain->spec->fill.fill_order_stride, chain->spec->tile_count});
            work_registration_orders.push_back({chain->spec->consume.consume_order_first,
                chain->spec->consume.consume_order_stride, chain->spec->tile_count});
        }
        for (size_t i = 0; i < work_id_ranges.size(); ++i) {
            for (size_t j = i + 1; j < work_id_ranges.size(); ++j) {
                const auto& a = work_id_ranges[i];
                const auto& b = work_id_ranges[j];
                if (a.first == b.first &&
                    a.second.first < b.second.second && b.second.first < a.second.second) {
                    fail(work.work_id, "compact virtual leg ID ranges overlap across descriptors");
                }
                const auto digit_extension = [](const std::string& short_prefix,
                                                const std::string& long_prefix) {
                    if (long_prefix.size() <= short_prefix.size() ||
                        long_prefix.compare(0, short_prefix.size(), short_prefix) != 0) {
                        return false;
                    }
                    for (size_t k = short_prefix.size(); k < long_prefix.size(); ++k) {
                        if (long_prefix[k] < '0' || long_prefix[k] > '9') return false;
                    }
                    return true;
                };
                if (a.first != b.first &&
                    (digit_extension(a.first, b.first) || digit_extension(b.first, a.first))) {
                    fail(work.work_id, "compact virtual leg ID prefixes have ambiguous decimal suffixes");
                }
            }
        }
        for (size_t i = 0; i < work_registration_orders.size(); ++i) {
            for (size_t j = i + 1; j < work_registration_orders.size(); ++j) {
                if (order_progressions_intersect(work_registration_orders[i],
                                                 work_registration_orders[j])) {
                    fail(work.work_id, "compact registration ordinals overlap across descriptors");
                }
            }
        }
        work_legs_.emplace(work.work_id, std::move(stored));
        work_specs_.emplace(work.work_id, &work);
        if (logical_count == 0) fail(work.work_id, "work has no logical legs");
        work_logical_leg_count_.emplace(work.work_id, logical_count);
        work_remaining_.emplace(work.work_id, logical_count);
        work_completed_leg_count_.emplace(work.work_id, 0);
        work_max_completion_.emplace(work.work_id, 0);
        work_done_at_.emplace(work.work_id, std::nullopt);
        deps_remaining_.emplace(std::string(kWorkNodePrefix) + work.work_id, 0);

        for (const auto& chain : pipeline_chains_) {
            if (chain->work_id == work.work_id) materialize_pipeline_pair(chain.get(), 0);
        }
        if (!work.pipeline_descriptors.empty()) {
            std::set<uint64_t> ordinary_ordinals;
            for (const auto* leg : work_legs_.at(work.work_id)) {
                if (!leg->has_registration_ordinal) {
                    fail(work.work_id, "descriptor-bearing ordinary leg lacks registration_ordinal");
                }
                if (!ordinary_ordinals.insert(leg->registration_ordinal).second) {
                    fail(work.work_id, "ordinary registration ordinals overlap");
                }
            }
            for (const auto& chain : pipeline_chains_) {
                if (chain->work_id != work.work_id) continue;
                const auto overlaps_ordinary = [&](uint64_t first, uint64_t stride) {
                    for (const auto ordinal : ordinary_ordinals) {
                        if (ordinal < first) continue;
                        const auto delta = ordinal - first;
                        if (delta % stride == 0 && delta / stride < chain->spec->tile_count) {
                            return true;
                        }
                    }
                    return false;
                };
                if (overlaps_ordinary(chain->spec->fill.fill_order_first,
                                      chain->spec->fill.fill_order_stride) ||
                    overlaps_ordinary(chain->spec->consume.consume_order_first,
                                      chain->spec->consume.consume_order_stride)) {
                    fail(work.work_id, "ordinary and compact pipeline registration ordinals overlap");
                }
            }
        }
        for (const auto* leg : work_legs_.at(work.work_id)) {
            if (find_virtual_ref(leg->spec->leg_id)) {
                fail(work.work_id, "ordinary leg ID aliases a compact virtual pipeline ID");
            }
        }
    }

    void wire_dependencies() {
        for (const auto& barrier : barriers_) {
            for (const auto& signaller : barrier.second.spec->signallers) {
                if (!legs_.count(signaller)) fail(barrier.first, "unknown signaller leg " + signaller);
            }
        }
        for (const auto& work : input_.background_works) {
            for (const auto& entry : legs_) {
                if (entry.second->work_id == work.work_id) {
                    register_leg_dependencies(entry.second.get());
                }
            }
            uint64_t count = 0;
            for (const auto& dep_work : work.depends_on_work) {
                if (!work_legs_.count(dep_work)) fail(work.work_id, "unknown work dependency " + dep_work);
                dependents_[std::string(kWorkNodePrefix) + dep_work].push_back(
                    std::string(kWorkNodePrefix) + work.work_id);
                ++count;
            }
            deps_remaining_[std::string(kWorkNodePrefix) + work.work_id] = count;
        }
        for (auto& entry : dependents_) std::sort(entry.second.begin(), entry.second.end());
        dependencies_wired_ = true;
    }

    void check_acyclic() const {
        auto indegree = deps_remaining_;
        // Future compact IDs are represented arithmetically. Their internal
        // chain order is acyclic by construction; exclude those incoming
        // edges from this bounded static/work-level cycle pass.
        for (const auto& entry : dependents_) {
            if (!find_virtual_ref(entry.first) || legs_.count(entry.first)) continue;
            for (const auto& target : entry.second) {
                const auto degree = indegree.find(target);
                if (degree == indegree.end() || degree->second == 0) {
                    fail(target, "dependency counter underflow in compact cycle check");
                }
                --degree->second;
            }
        }
        std::vector<std::string> queue;
        for (const auto& entry : indegree) if (entry.second == 0) queue.push_back(entry.first);
        size_t visited = 0;
        while (!queue.empty()) {
            auto node = std::move(queue.back());
            queue.pop_back();
            ++visited;
            const auto found = dependents_.find(node);
            if (found == dependents_.end()) continue;
            for (const auto& next : found->second) {
                auto& degree = indegree.at(next);
                if (--degree == 0) queue.push_back(next);
            }
        }
        if (visited != indegree.size()) fail("dependencies", "cycle in projection input");
    }

    void enter_service(RuntimeLeg* leg) {
        for (const auto& lease : leg->spec->acquires) {
            auto& gate = gates_.at(lease.gate_id);
            gate.occupied += lease.units;
            gate.max_occupied = std::max(gate.max_occupied, gate.occupied);
            if (gate.occupied > gate.spec->capacity_units) fail(leg->spec->leg_id, "in-service gate over capacity");
        }
        if (leg->remaining_bytes <= 0.0 && leg->remaining_duration <= 0.0) {
            complete(leg, now_ns_);
            return;
        }
        leg->state = kActive;
        active_add(leg);
        active_list_add(leg);
    }

    void push(double time_ns, int priority, std::string producer,
              int kind, std::string task_id) {
        events_.push({time_ns, priority, std::move(producer), sequence_++, kind,
                      std::move(task_id)});
    }

    uint64_t work_ready(const std::string& id) const {
        const auto found = work_specs_.find(id);
        return found == work_specs_.end()
            ? static_cast<uint64_t>(input_.now_ns) : found->second->ready_at_ns;
    }

    void on_deps_satisfied(RuntimeLeg* leg) {
        if (leg->state == kDone) return;
        const double ready = static_cast<double>(leg->spec->has_ready_at
            ? leg->spec->ready_at_ns : work_ready(leg->work_id));
        if (ready > now_ns_) {
            if (!leg->timer_scheduled) {
                leg->timer_scheduled = true;
                leg->state = kTimer;
                push(ready, 2, leg->spec->leg_id, 2, leg->spec->leg_id);
            }
            return;
        }
        try_activate(leg);
    }

    void try_activate(RuntimeLeg* leg) {
        for (const auto& lease : leg->spec->acquires) {
            const auto& gate = gates_.at(lease.gate_id);
            if (gate.occupied + lease.units > gate.spec->capacity_units) {
                auto& waiters = gates_.at(lease.gate_id).waiters;
                if (std::find(waiters.begin(), waiters.end(), leg) == waiters.end()) waiters.push_back(leg);
                leg->state = kGateWait;
                return;
            }
        }
        for (const auto& lease : leg->spec->acquires) {
            auto& gate = gates_.at(lease.gate_id);
            gate.occupied += lease.units;
            gate.max_occupied = std::max(gate.max_occupied, gate.occupied);
        }
        activate(leg);
    }

    void activate(RuntimeLeg* leg) {
        if (leg->spec->die_plan && !leg->die_execution) {
            leg->die_execution = std::make_unique<NativeDieExecution>(
                leg->spec->die_plan, contract_, effective_window_limit_);
            leg->die_phase_kind.clear();
            load_die_phase(leg, now_ns_);
        }
        if (leg->remaining_bytes <= 0.0 && leg->remaining_duration <= 0.0) {
            complete(leg, now_ns_);
            return;
        }
        if (leg->spec->latency_ns > 0) {
            leg->state = kLatency;
            push(now_ns_ + static_cast<double>(leg->spec->latency_ns), 1,
                 leg->spec->leg_id, 1, leg->spec->leg_id);
            return;
        }
        leg->state = kActive;
        active_add(leg);
        active_list_add(leg);
    }

    uint64_t active_members(const std::string& resource) const {
        const auto found = active_by_resource_.find(resource);
        return found == active_by_resource_.end() ? 0 : found->second.size();
    }

    void active_add(RuntimeLeg* leg) {
        if (leg->remaining_bytes > 0.0) {
            for (const auto& resource : leg->spec->route) {
                active_by_resource_[resource].insert(leg->spec->leg_id);
            }
        }
    }

    void active_list_add(RuntimeLeg* leg) {
        // Python's ordered active map is keyed by leg_id. A dependency
        // callback may activate a leg before the initial-ready scan reaches
        // it; the later scan cannot create a second active member.
        if (leg->in_active_list) return;
        leg->in_active_list = true;
        active_legs_.push_back(leg);
    }

    void active_remove(RuntimeLeg* leg) {
        for (const auto& resource : leg->spec->route) {
            auto found = active_by_resource_.find(resource);
            if (found != active_by_resource_.end()) found->second.erase(leg->spec->leg_id);
        }
        leg->in_active_list = false;
    }

    double rate_of(const RuntimeLeg* leg) const {
        double rate = std::numeric_limits<double>::infinity();
        for (const auto& key : leg->spec->route) {
            const double peak = resources_.at(key)->capacity_per_ns * eta_of(key);
            rate = std::min(rate, peak / static_cast<double>(
                std::max<uint64_t>(1, active_members(key))));
        }
        return rate;
    }

    std::map<std::string, FinishTimes> finishes() const {
        std::map<std::string, FinishTimes> result;
        for (const auto* leg : active_legs_) {
            std::vector<double> times;
            if (leg->remaining_bytes > 0.0) {
                times.push_back(now_ns_ + leg->remaining_bytes / rate_of(leg));
            }
            if (leg->remaining_duration > 0.0) {
                times.push_back(now_ns_ + leg->remaining_duration);
            }
            if (!times.empty()) {
                auto [minimum, maximum] = std::minmax_element(times.begin(), times.end());
                result.emplace(leg->spec->leg_id, FinishTimes{*minimum, *maximum});
            }
        }
        return result;
    }

    std::optional<double> next_boundary() const {
        const auto f = finishes();
        std::optional<double> earliest;
        for (const auto& entry : f) {
            if (!earliest || entry.second.boundary_contribution < *earliest) {
                earliest = entry.second.boundary_contribution;
            }
        }
        if (!events_.empty() && (!earliest || events_.top().time_ns < *earliest)) {
            earliest = events_.top().time_ns;
        }
        return earliest;
    }

    void integrate(double target_ns) {
        const double elapsed = target_ns - now_ns_;
        const double interval_start = now_ns_;
        if (elapsed < 0.0) fail("time", "integration moved backwards");
        double pre_remaining = 0.0;
        double served_total = 0.0;
        std::map<std::string, double> rates;
        std::vector<std::pair<std::string, double>> rates_order;
        const auto active = active_legs_; // preserve exact issue order
        for (const auto* leg : active) {
            if ((leg->spec->kind == "stream" || leg->spec->kind == "compute") &&
                leg->remaining_bytes > 0.0) {
                const double rate = rate_of(leg);
                rates.emplace(leg->spec->leg_id, rate);
                rates_order.emplace_back(leg->spec->leg_id, rate);
                pre_remaining += leg->remaining_bytes;
            }
        }
        for (auto* leg : active) {
            if (leg->remaining_duration > 0.0) {
                const double duration_finish = interval_start + leg->remaining_duration;
                leg->remaining_duration = duration_finish <= target_ns
                    ? 0.0 : leg->remaining_duration - elapsed;
            }
            const auto found = rates.find(leg->spec->leg_id);
            if (found != rates.end() && leg->remaining_bytes > 0.0) {
                const double rate = found->second;
                const double memory_finish = interval_start + leg->remaining_bytes / rate;
                const double served = rate * elapsed;
                leg->remaining_bytes -= served;
                served_total += served;
                bytes_served_total_ += served;
                if (memory_finish <= target_ns) {
                    leg->remaining_bytes = 0.0;
                } else if (leg->remaining_bytes < 0.0) {
                    const double dust = 1e-9 + rate * ulp(std::max(1.0, target_ns));
                    if (leg->remaining_bytes >= -dust) leg->remaining_bytes = 0.0;
                }
                if (leg->remaining_bytes <= 0.0) active_remove(leg);
            }
        }
        now_ns_ = target_ns;
        ++integration_steps_;
        if (!conservation_check_) return;
        // Mirror the frozen Python conservation checks in ordered double
        // arithmetic. These guard the port without changing any scheduling.
        double post_remaining = 0.0;
        for (const auto* leg : active_legs_) {
            if (leg->remaining_bytes > 0.0) post_remaining += leg->remaining_bytes;
        }
        double rate_sum = 0.0;
        for (const auto& entry : rates_order) rate_sum += entry.second;
        const double quant_slack = rate_sum * ulp(std::max(1.0, now_ns_));
        if (std::abs(pre_remaining - post_remaining - served_total) >
            (1e-6 * std::max(1.0, pre_remaining) + quant_slack)) {
            fail("conservation", "remaining decrease differs from served bytes (pre=" +
                 std::to_string(pre_remaining) + ", post=" +
                 std::to_string(post_remaining) + ", served=" +
                 std::to_string(served_total) + ", interval=" +
                 std::to_string(interval_start) + ".." +
                 std::to_string(now_ns_) + ")");
        }
        double expected = 0.0;
        for (const auto& entry : rates_order) expected += entry.second * elapsed;
        if (std::abs(served_total - expected) >
            (1e-6 * std::max(1.0, std::abs(expected)) + quant_slack)) {
            fail("conservation", "served bytes differ from rate*time");
        }
    }

    bool settle_completions(double time_ns,
                            const std::map<std::string, FinishTimes>& f,
                            uint64_t event_budget) {
        std::vector<RuntimeLeg*> done;
        for (const auto* leg : active_legs_) {
            const auto found = f.find(leg->spec->leg_id);
            if (found != f.end() && found->second.completion <= time_ns) done.push_back(const_cast<RuntimeLeg*>(leg));
        }
        std::sort(done.begin(), done.end(), [](const RuntimeLeg* a, const RuntimeLeg* b) {
            const int64_t ao = a->spec->has_order ? a->spec->order : 0;
            const int64_t bo = b->spec->has_order ? b->spec->order : 0;
            return std::tie(ao, a->spec->leg_id) < std::tie(bo, b->spec->leg_id);
        });
        for (auto* leg : done) {
            const uint64_t event_cost = leg->die_phase_kind == "timer" &&
                    leg->die_timer_uses_window
                ? leg->die_timer_logical_events : 1;
            if (event_budget != 0 &&
                (events_processed_ >= event_budget ||
                 event_cost > event_budget - events_processed_)) {
                last_step_budget_truncated_ = true;
                return false;
            }
            leg->remaining_bytes = 0.0;
            leg->remaining_duration = 0.0;
            if (event_cost > std::numeric_limits<uint64_t>::max() -
                                 events_processed_) {
                fail("events_processed", "logical projection event count overflows uint64");
            }
            events_processed_ += event_cost;
            complete(leg, time_ns);
        }
        return true;
    }

    bool process_events(double time_ns, uint64_t event_budget) {
        while (!events_.empty() && events_.top().time_ns <= time_ns) {
            if (event_budget != 0 && events_processed_ >= event_budget) {
                last_step_budget_truncated_ = true;
                return false;
            }
            const auto event = events_.top();
            events_.pop();
            ++events_processed_;
            if (event.kind == 1) {
                const auto found = legs_.find(event.task_id);
                if (found == legs_.end()) continue; // stale event for a retired virtual leg
                auto* leg = found->second.get();
                if (leg->state != kLatency) continue;
                leg->state = kActive;
                active_add(leg);
                active_list_add(leg);
            } else if (event.kind == 2) {
                const auto found = legs_.find(event.task_id);
                if (found == legs_.end()) continue; // stale event for a retired virtual leg
                auto* leg = found->second.get();
                if (leg->state != kTimer) continue;
                try_activate(leg);
            }
            // Forecast arrival is a disclosure-only placeholder.
        }
        return true;
    }

    std::optional<NativeBlockPipelineState> block_pipeline_state(
        const NativeProjectionBlockPipeline& pipeline,
        std::vector<RuntimeLeg*>* fills_out,
        std::vector<RuntimeLeg*>* consumes_out) const {
        const std::string work_id = "pipeline:" + pipeline.pipeline_id;
        const auto found = work_legs_.find(work_id);
        if (found == work_legs_.end()) return std::nullopt;
        std::vector<RuntimeLeg*> fills, consumes;
        for (auto* leg : found->second) {
            if (leg->spec->leg_id.find("#fill") != std::string::npos) fills.push_back(leg);
            else if (leg->spec->leg_id.find("#consume") != std::string::npos) consumes.push_back(leg);
        }
        if (fills.empty() || consumes.empty()) return std::nullopt;
        uint64_t done_fills = 0;
        for (const auto* fill : fills) if (fill->state == kDone) ++done_fills;
        const uint64_t remaining = static_cast<uint64_t>(fills.size()) - done_fills;
        if (remaining == 0) return std::nullopt;
        auto* next_fill = fills.at(static_cast<size_t>(done_fills));
        if (next_fill->state != kActive) return std::nullopt;
        if (std::any_of(fills.begin(), fills.end(), [](const auto* fill) {
                return fill->state == kGateWait;
            })) return std::nullopt;
        double consuming = 0.0;
        uint64_t buffered = 0;
        for (size_t index = 0; index < consumes.size(); ++index) {
            const auto* consume = consumes[index];
            if (consume->state == kDone) continue;
            if (consume->state == kActive) consuming = consume->remaining_duration;
            else if (fills[index]->state == kDone) ++buffered;
        }
        NativeBlockPipelineState state;
        state.now_ns = now_ns_;
        state.rate_bns = rate_of(next_fill);
        state.blocks_remaining = remaining;
        state.in_fill_served_bytes = static_cast<double>(pipeline.block_bytes) -
            next_fill->remaining_bytes;
        state.buffered_full_blocks = buffered;
        state.consuming_remaining_ns = consuming;
        if (fills_out) *fills_out = std::move(fills);
        if (consumes_out) *consumes_out = std::move(consumes);
        return state;
    }

    NativeBlockPipelinePlan block_pipeline_plan(
        const NativeBlockPipelineState& state,
        const NativeProjectionBlockPipeline& spec) const {
        NativeBlockPipelinePlan plan;
        std::vector<double> release_times;
        double cursor = state.now_ns + state.consuming_remaining_ns;
        const uint64_t old_count = state.buffered_full_blocks +
            (state.consuming_remaining_ns > 0.0 ? 1 : 0);
        for (uint64_t index = 0; index < old_count; ++index) {
            release_times.push_back(cursor);
            cursor += spec.consume_duration_ns;
        }
        std::optional<double> last_consume_end = release_times.empty()
            ? std::nullopt : std::optional<double>(release_times.back());
        std::optional<double> previous_fill_end;
        for (uint64_t index = 0; index < state.blocks_remaining; ++index) {
            const double served = index == 0 ? state.in_fill_served_bytes : 0.0;
            const double fill_time = (static_cast<double>(spec.block_bytes) - served) /
                state.rate_bns;
            double start = state.now_ns;
            if (index != 0) {
                std::optional<double> gate_time;
                if (release_times.size() >= spec.buffer_slots) {
                    auto ordered = release_times;
                    std::sort(ordered.begin(), ordered.end());
                    gate_time = ordered[ordered.size() -
                        static_cast<size_t>(spec.buffer_slots)];
                }
                start = std::max(previous_fill_end.value_or(state.now_ns),
                                 gate_time.value_or(state.now_ns));
            }
            const double fill_end = start + fill_time;
            const double consume_start = std::max(
                fill_end, last_consume_end.value_or(fill_end));
            const double consume_end = consume_start + spec.consume_duration_ns;
            release_times.push_back(consume_end);
            last_consume_end = consume_end;
            previous_fill_end = fill_end;
            plan.fill_starts.push_back(start);
            plan.fill_ends.push_back(fill_end);
            plan.consume_ends.push_back(consume_end);
        }
        plan.finish = last_consume_end.value_or(state.now_ns);
        for (uint64_t index = 0; index < old_count && index < release_times.size(); ++index) {
            if (release_times[index] > state.now_ns) {
                plan.old_gate_release_times.push_back(release_times[index]);
            }
        }
        return plan;
    }

    bool try_analytic_jump() {
        for (const auto& pipeline : input_.pipelines) {
            const std::string work_id = "pipeline:" + pipeline.pipeline_id;
            std::set<std::string> members;
            const auto work = work_legs_.find(work_id);
            if (work == work_legs_.end()) continue;
            for (const auto* leg : work->second) members.insert(leg->spec->leg_id);
            if (members.empty()) continue;
            bool external_stream = false;
            for (const auto* leg : active_legs_) {
                if (leg->remaining_bytes <= 0.0 || members.count(leg->spec->leg_id)) continue;
                if (std::find(leg->spec->route.begin(), leg->spec->route.end(),
                              pipeline.fill_resource) != leg->spec->route.end()) {
                    external_stream = true;
                    break;
                }
            }
            if (external_stream) continue;
            const auto& gate = gates_.at(pipeline.buffer_gate_id);
            if (std::any_of(gate.waiters.begin(), gate.waiters.end(), [&](const auto* leg) {
                    return !members.count(leg->spec->leg_id);
                })) continue;
            bool external_dependent = false;
            std::vector<std::string> sources(members.begin(), members.end());
            sources.push_back(std::string(kWorkNodePrefix) + work_id);
            for (const auto& node : sources) {
                const auto deps = dependents_.find(node);
                if (deps == dependents_.end()) continue;
                if (std::any_of(deps->second.begin(), deps->second.end(), [&](const auto& id) {
                        return !members.count(id);
                    })) {
                    external_dependent = true;
                    break;
                }
            }
            if (external_dependent) continue;
            std::vector<RuntimeLeg*> fills, consumes;
            const auto state = block_pipeline_state(pipeline, &fills, &consumes);
            if (!state) continue;
            const auto plan = block_pipeline_plan(*state, pipeline);
            std::optional<double> external_boundary;
            const auto current_finishes = finishes();
            for (const auto& entry : current_finishes) {
                if (members.count(entry.first)) continue;
                if (!external_boundary || entry.second.boundary_contribution < *external_boundary) {
                    external_boundary = entry.second.boundary_contribution;
                }
            }
            auto pending_events = events_;
            while (!pending_events.empty()) {
                const auto event = pending_events.top();
                pending_events.pop();
                if (!members.count(event.task_id) &&
                    (!external_boundary || event.time_ns < *external_boundary)) {
                    external_boundary = event.time_ns;
                }
            }
            if (external_boundary && plan.finish > *external_boundary) continue;
            std::set<double> event_times(plan.fill_ends.begin(), plan.fill_ends.end());
            event_times.insert(plan.consume_ends.begin(), plan.consume_ends.end());
            event_times.insert(plan.old_gate_release_times.begin(), plan.old_gate_release_times.end());
            for (const double event_time : event_times) {
                const auto event_finishes = finishes();
                integrate(event_time);
                (void)settle_completions(event_time, event_finishes, 0);
                (void)process_events(event_time, 0);
            }
            std::vector<std::string> ordered_members(members.begin(), members.end());
            std::sort(ordered_members.begin(), ordered_members.end());
            for (const auto& id : ordered_members) {
                const auto found = legs_.find(id);
                if (found == legs_.end() || found->second->state == kDone) continue;
                found->second->remaining_bytes = 0.0;
                found->second->remaining_duration = 0.0;
                ++events_processed_;
                complete(found->second.get(), plan.finish);
            }
            return true;
        }
        return false;
    }

    bool step(uint64_t event_budget = 0) {
        last_step_budget_truncated_ = false;
        if (analytic_enabled_ && event_budget == 0 && try_analytic_jump()) return true;
        const auto f = finishes();
        std::vector<double> candidates;
        for (const auto& entry : f) candidates.push_back(entry.second.boundary_contribution);
        if (!events_.empty()) candidates.push_back(events_.top().time_ns);
        if (candidates.empty()) return false;
        const double boundary = *std::min_element(candidates.begin(), candidates.end());
        integrate(boundary);
        if (!settle_completions(boundary, f, event_budget)) return true;
        if (!process_events(boundary, event_budget)) return true;
        return true;
    }

    void wake_gate_waiters(RuntimeGate& gate) {
        bool progressed = true;
        while (progressed) {
            progressed = false;
            size_t index = 0;
            while (index < gate.waiters.size()) {
                auto* leg = gate.waiters[index];
                bool fits = true;
                for (const auto& lease : leg->spec->acquires) {
                    const auto& other = gates_.at(lease.gate_id);
                    if (other.occupied + lease.units > other.spec->capacity_units) {
                        fits = false;
                        break;
                    }
                }
                if (fits) {
                    gate.waiters.erase(gate.waiters.begin() + static_cast<std::ptrdiff_t>(index));
                    try_activate(leg);
                    progressed = true;
                    break;
                }
                ++index;
            }
        }
    }

    void complete(RuntimeLeg* leg, double time_ns) {
        if (leg->spec->die_plan && leg->die_phase_kind != "done") {
            if (!leg->die_execution) fail(leg->spec->leg_id, "die phase completed without execution");
            if (leg->die_phase_kind == "timer") {
                const uint64_t service_delta = round_ties_even_u64(
                    leg->die_phase_ns, leg->spec->leg_id + ".timer_delta");
                if (leg->die_timer_uses_window) {
                    leg->die_execution->advance_service_window(
                        leg->die_timer_window_token, service_delta);
                    leg->die_execution->record_timer_callback(
                        leg->die_timer_logical_events,
                        &leg->die_timer_window_plan);
                } else {
                    leg->die_execution->advance_time(service_delta);
                    if (leg->die_timer_has_window_plan) {
                        leg->die_execution->record_timer_callback(
                            leg->die_timer_logical_events,
                            &leg->die_timer_window_plan);
                    } else {
                        leg->die_execution->record_timer_callback(
                            leg->die_timer_logical_events);
                    }
                }
            } else if (leg->die_phase_kind == "memory") {
                leg->die_execution->advance_time(round_ties_even_u64(
                    time_ns - leg->die_phase_ns, leg->spec->leg_id + ".memory_delta"));
                leg->die_execution->complete_memory(leg->die_stage_id);
            } else {
                fail(leg->spec->leg_id, "die phase settled without a timer or memory request");
            }
            load_die_phase(leg, time_ns);
            if (leg->die_phase_kind != "done") {
                leg->state = kActive;
                active_add(leg);
                return;
            }
        }
        leg->state = kDone;
        auto active_it = std::find(active_legs_.begin(), active_legs_.end(), leg);
        if (active_it != active_legs_.end()) {
            active_remove(leg);
            active_legs_.erase(active_it);
        }
        leg->has_completion = true;
        leg->completion = ceil_ns(time_ns);
        const std::string completed_id = leg->spec->leg_id;
        const std::string completed_work_id = leg->work_id;
        RuntimePipelineChain* pipeline_chain = leg->pipeline_chain;
        const bool pipeline_fill = leg->pipeline_fill;
        const uint64_t pipeline_index = leg->pipeline_local_index;
        for (const auto& lease : leg->spec->releases) {
            auto& gate = gates_.at(lease.gate_id);
            if (gate.occupied < lease.units) fail(gate.spec->gate_id, "released below zero");
            gate.occupied -= lease.units;
            wake_gate_waiters(gate);
        }
        if (pipeline_chain != nullptr) {
            if (pipeline_fill) {
                if (pipeline_index != pipeline_chain->completed_fill_count) {
                    fail(completed_id, "fill cursor completed out of sequence");
                }
                // Register the next tile's dependency while this predecessor
                // is still logically uncompleted. signal_dependents() below
                // then wakes it in the same lexical callback order as the
                // original fully materialized graph.
                if (pipeline_index + 1 < pipeline_chain->spec->tile_count) {
                    materialize_pipeline_pair(pipeline_chain, pipeline_index + 1);
                }
                ++pipeline_chain->completed_fill_count;
            } else {
                if (pipeline_index != pipeline_chain->completed_consume_count) {
                    fail(completed_id, "consume cursor completed out of sequence");
                }
                ++pipeline_chain->completed_consume_count;
                if (pipeline_index + 1 == pipeline_chain->spec->tile_count) {
                    pipeline_chain->has_terminal_consume_completion = true;
                    pipeline_chain->terminal_consume_completion = leg->completion;
                }
            }
        }
        if (leg->spec->has_signal_barrier) {
            auto& barrier = barriers_.at(leg->spec->signals_barrier);
            barrier.arrived.insert(leg->spec->leg_id);
            if (barrier.arrived.size() == barrier.spec->signallers.size()) {
                barrier.has_passed_at = true;
                barrier.passed_at = ceil_ns(time_ns);
                signal_dependents(leg->spec->signals_barrier);
            }
        }
        signal_dependents(completed_id);
        auto& remaining = work_remaining_.at(completed_work_id);
        if (remaining == 0) fail(completed_work_id, "work completion underflow");
        --remaining;
        ++work_completed_leg_count_.at(completed_work_id);
        work_max_completion_.at(completed_work_id) = std::max(
            work_max_completion_.at(completed_work_id), leg->completion);
        if (remaining == 0) {
            work_done_at_[completed_work_id] = work_max_completion_.at(completed_work_id);
            signal_dependents(std::string(kWorkNodePrefix) + completed_work_id);
        }
        if (pipeline_chain != nullptr) {
            dependents_.erase(completed_id);
            deps_remaining_.erase(completed_id);
            legs_.erase(completed_id);
        }
    }

    void load_die_phase(RuntimeLeg* leg, double at_time_ns) {
        while (true) {
            const auto action = leg->die_execution->next_action();
            if (action.kind == NativeDieAction::Kind::Done) {
                leg->die_phase_kind = "done";
                leg->die_execution->close();
                leg->remaining_bytes = 0.0;
                leg->remaining_duration = 0.0;
                return;
            }
            if (action.kind == NativeDieAction::Kind::Timer) {
                if (action.ns == 0) {
                    leg->die_execution->advance_time(0);
                    continue;
                }
                leg->die_phase_kind = "timer";
                leg->die_phase_ns = static_cast<double>(action.ns);
                leg->remaining_bytes = 0.0;
                leg->remaining_duration = static_cast<double>(action.ns);
                leg->die_timer_logical_events = 1;
                leg->die_timer_uses_window = false;
                leg->die_timer_has_window_plan = false;
                leg->die_timer_window_plan = PipelineWindowPlan{};
                leg->die_timer_window_token =
                    DieModel::InternalServiceWindowToken{};
                const uint32_t limit = remaining_window_limit();
                if (limit > 1) {
                    const NativeDieTimerWindow selected =
                        leg->die_execution->prepare_timer_window(limit);
                    if (selected.uses_service_window &&
                        has_external_boundary_through(
                            leg, at_time_ns + selected.external_delay_ns)) {
                        leg->die_execution->reject_timer_window(
                            selected,
                            "external_projection_boundary_inside_window");
                    } else if (selected.uses_service_window) {
                        leg->die_phase_ns = static_cast<double>(
                            selected.original_service_ns);
                        leg->remaining_duration = static_cast<double>(
                            selected.external_delay_ns);
                        leg->die_timer_logical_events =
                            selected.plan.selected_count;
                        leg->die_timer_uses_window = true;
                        leg->die_timer_has_window_plan = true;
                        leg->die_timer_window_plan = selected.plan;
                        leg->die_timer_window_token = selected.token;
                    } else if (selected.has_exact_feedback) {
                        leg->die_timer_has_window_plan = true;
                        leg->die_timer_window_plan = selected.plan;
                    }
                }
                return;
            }
            if (action.kind == NativeDieAction::Kind::Memory) {
                if (!(action.service_units >= 0.0) || !std::isfinite(action.service_units) ||
                    !std::isfinite(action.actual_bytes) || action.actual_bytes < 0.0 ||
                    static_cast<long double>(action.actual_bytes) >
                        static_cast<long double>(std::numeric_limits<uint64_t>::max())) {
                    fail(leg->spec->leg_id, "invalid DieModel memory action quantities");
                }
                if (action.service_units <= 0.0) {
                    leg->die_execution->advance_time(0);
                    leg->die_execution->complete_memory(action.stage_id);
                    continue;
                }
                leg->die_phase_kind = "memory";
                leg->die_timer_logical_events = 1;
                leg->die_timer_uses_window = false;
                leg->die_timer_has_window_plan = false;
                leg->die_stage_id = action.stage_id;
                leg->die_phase_ns = at_time_ns;
                leg->remaining_duration = 0.0;
                leg->remaining_bytes = action.service_units;
                const auto actual = static_cast<uint64_t>(action.actual_bytes);
                if (actual > std::numeric_limits<uint64_t>::max() - leg->die_actual_bytes_total) {
                    fail(leg->spec->leg_id, "die actual-byte disclosure overflows uint64");
                }
                leg->die_actual_bytes_total += actual;
                leg->die_service_units_total += action.service_units;
                return;
            }
        }
    }

    void signal_dependents(const std::string& node) {
        const auto found = dependents_.find(node);
        if (found == dependents_.end()) return;
        for (const auto& leg_id : found->second) {
            auto& count = deps_remaining_.at(leg_id);
            if (count == 0) fail(leg_id, "dependency counter underflow");
            --count;
            if (count == 0) {
                const auto target = legs_.find(leg_id);
                if (target != legs_.end()) on_deps_satisfied(target->second.get());
                else if (!has_prefix(leg_id, kWorkNodePrefix)) {
                    fail(leg_id, "dependency target is not registered");
                }
            }
        }
    }
};

uint64_t projection_share_units_in_chain(
    const PipelineProjectionShare& share, const uint64_t tile_count,
    const std::string& path) {
    const unsigned __int128 end =
        static_cast<unsigned __int128>(share.share_index_first) + tile_count;
    if (tile_count == 0 || end > share.total_tiles) {
        fail(path, "chain share range is outside total_tiles");
    }
    const uint64_t quotient = share.total_units / share.total_tiles;
    const uint64_t remainder = share.total_units % share.total_tiles;
    const unsigned __int128 base =
        static_cast<unsigned __int128>(quotient) * tile_count;
    const uint64_t overlap_start = std::min(share.share_index_first, remainder);
    const uint64_t overlap_end = std::min(
        static_cast<uint64_t>(end), remainder);
    const uint64_t remainder_units = overlap_end > overlap_start
        ? overlap_end - overlap_start : 0;
    const unsigned __int128 total = base + remainder_units;
    if (total > std::numeric_limits<uint64_t>::max()) {
        fail(path, "chain share sum exceeds uint64");
    }
    return static_cast<uint64_t>(total);
}

uint64_t projection_fill_bytes_in_chain(
    const PipelineDescriptor& descriptor,
    const PipelineProjectionChain& chain) {
    const auto group = std::find_if(
        descriptor.groups().begin(), descriptor.groups().end(),
        [&](const auto& candidate) {
            return candidate.group_ordinal == chain.group_ordinal;
        });
    if (group == descriptor.groups().end()) {
        fail("focus_bytes", "pipeline chain references missing group");
    }
    if (chain.tile_count == 0 || chain.tile_count != group->tile_axis.tile_count) {
        fail("focus_bytes", "pipeline chain tile count disagrees with group axis");
    }
    const auto& axis = group->tile_axis;
    const unsigned __int128 prior_units =
        static_cast<unsigned __int128>(axis.units_per_tile) *
        (chain.tile_count - 1);
    if (prior_units > axis.total_units) {
        fail("focus_bytes", "pipeline group tile extent is inconsistent");
    }
    const uint64_t tail_remaining = axis.total_units -
        static_cast<uint64_t>(prior_units);
    const uint64_t tail_bytes = std::min(
        chain.fill.bytes_per_tile, tail_remaining);
    const unsigned __int128 total =
        static_cast<unsigned __int128>(chain.fill.bytes_per_tile) *
            (chain.tile_count - 1) + tail_bytes;
    if (total > std::numeric_limits<uint64_t>::max()) {
        fail("focus_bytes", "pipeline fill total exceeds uint64");
    }
    return static_cast<uint64_t>(total);
}

}  // namespace

namespace {

struct ClosureSummary final {
    std::set<std::string> resources;
    std::set<std::string> works;
};

ClosureSummary compute_closure(const NativeProjectionInput& input,
                               const std::vector<NativeProjectionWorkSpec>& candidates) {
    std::vector<const NativeProjectionWorkSpec*> works;
    works.reserve(input.background_works.size() + candidates.size());
    const std::set<std::string> explicit_work_ids(
        input.explicit_background_work_ids.begin(),
        input.explicit_background_work_ids.end());
    for (const auto& work : input.background_works) {
        if (explicit_work_ids.count(work.work_id)) works.push_back(&work);
    }
    for (const auto& work : candidates) works.push_back(&work);
    std::map<std::string, std::set<std::string>> resource_to_works;
    std::map<std::string, std::set<std::string>> gate_to_works;
    std::map<std::string, std::set<std::string>> barrier_to_works;
    std::map<std::string, std::set<std::string>> dep_edges;
    std::map<std::string, std::string> leg_to_work;
    struct VirtualOwner final { std::string prefix, work_id; uint64_t first, count; };
    std::vector<VirtualOwner> virtual_owners;
    std::vector<std::pair<std::string, std::string>> compact_dependencies;
    std::set<std::string> candidate_ids;
    for (const auto* work : works) {
        for (const auto& leg : work->legs) leg_to_work[leg.leg_id] = work->work_id;
        for (const auto& descriptor : work->pipeline_descriptors) {
            for (const auto& chain : descriptor->projection().chains) {
                virtual_owners.push_back({chain.fill.leg_id_prefix, work->work_id,
                    chain.fill.first_index, chain.tile_count});
                virtual_owners.push_back({chain.consume.leg_id_prefix, work->work_id,
                    chain.consume.first_index, chain.tile_count});
                for (const auto& key : chain.fill.route) resource_to_works[key].insert(work->work_id);
                for (const auto& key : chain.consume.route) resource_to_works[key].insert(work->work_id);
                gate_to_works[chain.fill.gate_id].insert(work->work_id);
                for (const auto& dep : chain.fill.start_deps) {
                    compact_dependencies.emplace_back(work->work_id, dep);
                }
                for (const auto& dep : chain.consume.consume_deps) {
                    compact_dependencies.emplace_back(work->work_id, dep);
                }
                for (const auto& dep : chain.consume.consume_start_deps) {
                    compact_dependencies.emplace_back(work->work_id, dep);
                }
            }
        }
    }
    for (const auto& work : candidates) candidate_ids.insert(work.work_id);
    const auto virtual_owner = [&](const std::string& id) -> std::optional<std::string> {
        std::optional<std::string> owner;
        for (const auto& range : virtual_owners) {
            if (id.size() <= range.prefix.size() ||
                id.compare(0, range.prefix.size(), range.prefix) != 0) continue;
            const auto suffix = id.substr(range.prefix.size());
            uint64_t index = 0;
            bool valid = !suffix.empty();
            for (char ch : suffix) {
                if (ch < '0' || ch > '9') { valid = false; break; }
                const uint64_t digit = static_cast<uint64_t>(ch - '0');
                if (index > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
                    valid = false; break;
                }
                index = index * 10 + digit;
            }
            if (!valid || std::to_string(index) != suffix || index < range.first ||
                index - range.first >= range.count) continue;
            if (owner && *owner != range.work_id) {
                fail(id, "compact dependency resolves to multiple work owners");
            }
            owner = range.work_id;
        }
        return owner;
    };
    const auto add_dependency_edge = [&](const std::string& work_id,
                                         const std::string& dep) {
        auto other = leg_to_work.find(dep);
        if (other != leg_to_work.end()) {
            dep_edges[work_id].insert(other->second);
            dep_edges[other->second].insert(work_id);
        } else if (const auto compact_owner = virtual_owner(dep)) {
            dep_edges[work_id].insert(*compact_owner);
            dep_edges[*compact_owner].insert(work_id);
        } else if (has_prefix(dep, kWorkNodePrefix)) {
            const auto target = dep.substr(std::char_traits<char>::length(kWorkNodePrefix));
            dep_edges[work_id].insert(target);
            dep_edges[target].insert(work_id);
        } else {
            barrier_to_works[dep].insert(work_id);
        }
    };
    for (const auto* work : works) {
        for (const auto& leg : work->legs) {
            for (const auto& key : leg.route) resource_to_works[key].insert(work->work_id);
            for (const auto& lease : leg.acquires) gate_to_works[lease.gate_id].insert(work->work_id);
            for (const auto& lease : leg.releases) gate_to_works[lease.gate_id].insert(work->work_id);
            if (leg.has_signal_barrier) barrier_to_works[leg.signals_barrier].insert(work->work_id);
            for (const auto& dep : leg.deps) {
                add_dependency_edge(work->work_id, dep);
            }
        }
        for (const auto& compact : compact_dependencies) {
            if (compact.first == work->work_id) add_dependency_edge(work->work_id, compact.second);
        }
        for (const auto& dep : work->depends_on_work) {
            dep_edges[work->work_id].insert(dep);
            dep_edges[dep].insert(work->work_id);
        }
    }
    for (const auto& barrier : input.barriers) {
        for (const auto& id : barrier.signallers) {
            auto owner = leg_to_work.find(id);
            if (owner != leg_to_work.end()) barrier_to_works[barrier.barrier_id].insert(owner->second);
        }
    }
    ClosureSummary out;
    out.works = std::move(candidate_ids);
    std::set<std::string> gates, barriers;
    for (const auto& work : candidates) {
        for (const auto& leg : work.legs) {
            for (const auto& dep : leg.deps) {
                for (const auto& barrier : input.barriers) {
                    if (barrier.barrier_id == dep) barriers.insert(dep);
                }
            }
        }
    }
    for (const auto& compact : compact_dependencies) {
        if (candidate_ids.count(compact.first)) {
            for (const auto& barrier : input.barriers) {
                if (barrier.barrier_id == compact.second) barriers.insert(compact.second);
            }
        }
    }
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto* work : works) {
            if (!out.works.count(work->work_id)) continue;
            for (const auto& leg : work->legs) {
                for (const auto& key : leg.route) changed |= out.resources.insert(key).second;
                for (const auto& lease : leg.acquires) changed |= gates.insert(lease.gate_id).second;
                for (const auto& lease : leg.releases) changed |= gates.insert(lease.gate_id).second;
                if (leg.has_signal_barrier) changed |= barriers.insert(leg.signals_barrier).second;
            }
            for (const auto& descriptor : work->pipeline_descriptors) {
                for (const auto& chain : descriptor->projection().chains) {
                    for (const auto& key : chain.fill.route) changed |= out.resources.insert(key).second;
                    for (const auto& key : chain.consume.route) changed |= out.resources.insert(key).second;
                    changed |= gates.insert(chain.fill.gate_id).second;
                }
            }
        }
        auto add_index = [&](const auto& index, const std::set<std::string>& closed) {
            for (const auto& key : closed) {
                const auto found = index.find(key);
                if (found == index.end()) continue;
                for (const auto& work_id : found->second) changed |= out.works.insert(work_id).second;
            }
        };
        add_index(resource_to_works, out.resources);
        add_index(gate_to_works, gates);
        add_index(barrier_to_works, barriers);
        const auto before = out.works;
        for (const auto& id : before) {
            const auto found = dep_edges.find(id);
            if (found != dep_edges.end()) {
                for (const auto& other : found->second) changed |= out.works.insert(other).second;
            }
        }
    }
    return out;
}

Json background_comparison(const std::string& id,
                           const std::optional<uint64_t>& before,
                           const std::optional<uint64_t>& after) {
    Json before_json = before ? Json(*before) : Json(nullptr);
    Json after_json = after ? Json(*after) : Json(nullptr);
    Json delay = (before && after) ? Json(static_cast<int64_t>(*after) - static_cast<int64_t>(*before)) : Json(nullptr);
    return Json{{"work_id", id}, {"done_without_candidate_ns", before_json},
                {"done_with_candidate_ns", after_json}, {"delay_ns", delay}};
}

Json quality_fields(const std::string& status, uint64_t events,
                    uint64_t steps, const std::string& unsupported = "") {
    return Json{{"approximation_profile", "exact"}, {"window_count", 0},
                {"refinement_count", 0}, {"local_prefix_bound_max", 0},
                {"logical_event_count", events}, {"physical_step_count", steps},
                {"bound_kind", "none"},
                {"fully_exact", status != kStatusDegraded && status != kStatusUnavailable},
                {"unsupported_reason", unsupported.empty() ? Json(nullptr) : Json(unsupported)}};
}

}  // namespace

NativeProjectionContext::NativeProjectionContext(
    std::shared_ptr<const NativeProjectionInput> input,
    std::shared_ptr<const PipelineKernelContract> contract,
    Json kernel_contract)
    : input_(std::move(input)), contract_(std::move(contract)),
      kernel_contract_(std::move(kernel_contract)) {}

std::shared_ptr<const NativeProjectionContext> NativeProjectionContext::parse(
    const nlohmann::json& snapshot_envelope) {
    object(snapshot_envelope, "snapshot_envelope", {"projection_input"},
           {"kernel_contract"});
    std::shared_ptr<const PipelineKernelContract> contract;
    Json kernel_contract = nullptr;
    if (snapshot_envelope.contains("kernel_contract")) {
        contract = PipelineKernelContract::parse(
            snapshot_envelope.at("kernel_contract"));
        kernel_contract = snapshot_envelope.at("kernel_contract");
    }
    auto input = parse_projection_input(snapshot_envelope.at("projection_input"));
    return std::shared_ptr<const NativeProjectionContext>(
        new NativeProjectionContext(std::move(input), std::move(contract),
                                    std::move(kernel_contract)));
}

const NativeProjectionContext::ReferenceRunResult*
NativeProjectionContext::lookup_reference_run(uint64_t key) const {
    const auto found = reference_cache_index_.find(key);
    if (found == reference_cache_index_.end()) {
        ++reference_cache_misses_;
        return nullptr;
    }
    // Move to front (LRU freshness).
    reference_cache_.splice(reference_cache_.begin(), reference_cache_,
                            found->second);
    found->second = reference_cache_.begin();
    ++reference_cache_hits_;
    return &found->second->second;
}

const NativeProjectionContext::ReferenceRunResult*
NativeProjectionContext::store_reference_run(
    uint64_t key, ReferenceRunResult result) const {
    const auto existing = reference_cache_index_.find(key);
    if (existing != reference_cache_index_.end()) {
        existing->second->second = std::move(result);
        return &existing->second->second;
    }
    if (reference_cache_.size() >= kReferenceCacheCapacity) {
        const auto lru = std::prev(reference_cache_.end());
        reference_cache_index_.erase(lru->first);
        reference_cache_.erase(lru);
    }
    reference_cache_.emplace_front(key, std::move(result));
    reference_cache_index_.emplace(key, reference_cache_.begin());
    return &reference_cache_.front().second;
}

NativeProjectionOptions NativeProjectionOptions::parse(
    const nlohmann::json& wire, const nlohmann::json& kernel_contract) {
    object(wire, "options", {}, {"projection_event_budget",
        "closure_resource_budget", "horizon_ns", "include_background_reference",
        "analytic_segments", "drain_background", "effective_window_tile_limit"});
    NativeProjectionOptions result;
    if (wire.contains("projection_event_budget")) {
        result.projection_event_budget = uint_value(wire.at("projection_event_budget"), "options.projection_event_budget");
    }
    if (wire.contains("closure_resource_budget")) {
        result.closure_resource_budget = uint_value(wire.at("closure_resource_budget"), "options.closure_resource_budget");
    }
    if (wire.contains("horizon_ns") && !wire.at("horizon_ns").is_null()) {
        result.horizon_ns = number_value(wire.at("horizon_ns"), "options.horizon_ns");
        if (result.horizon_ns < 0.0) fail("options.horizon_ns", "must be nonnegative");
        result.has_horizon = true;
    }
    for (const auto& item : {std::pair<const char*, bool*>("include_background_reference", &result.include_background_reference),
                             std::pair<const char*, bool*>("analytic_segments", &result.analytic_segments),
                             std::pair<const char*, bool*>("drain_background", &result.drain_background)}) {
        if (wire.contains(item.first)) {
            if (!wire.at(item.first).is_boolean()) fail(std::string("options.") + item.first, "expected boolean");
            *item.second = wire.at(item.first).get<bool>();
        }
    }
    const bool has_contract = kernel_contract.is_object();
    const uint64_t max_k = has_contract
        ? uint_value(kernel_contract.at("window_tile_limit"), "kernel_contract.window_tile_limit")
        : 1;
    const uint64_t requested = wire.contains("effective_window_tile_limit")
        ? uint_value(wire.at("effective_window_tile_limit"), "options.effective_window_tile_limit")
        : max_k;
    if (!(requested == 1 || requested == 2 || requested == 4 || requested == 8) || requested > max_k) {
        fail("options.effective_window_tile_limit",
             has_contract
                 ? "must be a supported limit no greater than the frozen run contract"
                 : "standalone context without a frozen kernel_contract is exact-only at K=1");
    }
    result.effective_window_tile_limit = static_cast<uint32_t>(requested);
    return result;
}

nlohmann::json NativeProjection::project(
    const NativeProjectionContext& context,
    const nlohmann::json& candidate_payload,
    const NativeProjectionOptions& options) {
    // Payload contract (v5 shared-snapshot form): an object carrying the
    // bounded candidate WorkItems plus the per-call private overlay of the
    // candidate-declared resources/gates/barriers and §7.2 inherited
    // obligation works that the legacy per-candidate snapshot baked into
    // its creation input.
    if (!candidate_payload.is_object()) {
        fail("candidate_payload", "expected an object with 'candidates'");
    }
    object(candidate_payload, "candidate_payload", {"candidates"},
           {"private_overlay"});
    const Json& candidate_work_items = candidate_payload.at("candidates");
    if (!candidate_work_items.is_array()) fail("candidates", "expected an array of WorkItem objects");
    const Json overlay_wire = candidate_payload.contains("private_overlay")
        ? candidate_payload.at("private_overlay") : Json(Json::object());
    const NativeProjectionOverlay overlay =
        parse_projection_overlay(overlay_wire);
    NativeProjectionInput merged_base =
        apply_projection_overlay(context.input(), overlay);
    std::vector<NativeProjectionWorkSpec> candidates;
    std::set<std::string> candidate_ids;
    std::set<std::string> base_ids;
    std::set<std::string> leg_ids;
    for (const auto& work : merged_base.background_works) {
        base_ids.insert(work.work_id);
        for (const auto& leg : work.legs) leg_ids.insert(leg.leg_id);
    }
    candidates.reserve(candidate_work_items.size());
    for (size_t i = 0; i < candidate_work_items.size(); ++i) {
        const auto path = "candidates[" + std::to_string(i) + "]";
        auto work = parse_work_spec(candidate_work_items[i], path, 1);
        if (base_ids.count(work.work_id) || !candidate_ids.insert(work.work_id).second) {
            fail(path + ".work_id", "duplicate candidate work id");
        }
        for (const auto& leg : work.legs) {
            if (!leg_ids.insert(leg.leg_id).second) fail(path + ".legs", "duplicate leg id");
        }
        candidates.push_back(std::move(work));
    }
    const ClosureSummary closure = compute_closure(merged_base, candidates);
    const uint32_t effective_window_limit =
        options.effective_window_tile_limit == 0
            ? (context.has_kernel_contract()
                   ? context.contract().window_tile_limit() : 1U)
            : options.effective_window_tile_limit;

    // The reference world is a pure function of the shared snapshot plus the
    // reference-relevant overlay fields (gates, extra background works,
    // added resources) and the run options; the candidate set never enters
    // it (candidates live outside the snapshot, so the candidate-id filter
    // below is retained for exactness but cannot remove anything).  Within
    // one context, candidates sharing those inputs reuse the memoized run.
    // Running the reference before the candidate world does not change any
    // observable result: both are deterministic simulations of independent
    // inputs.
    const std::set<std::string> explicit_background_ids(
        merged_base.explicit_background_work_ids.begin(),
        merged_base.explicit_background_work_ids.end());
    // Capture comparison ids (not pointers): the reference-miss path below
    // consumes merged_base by move, invalidating any pointers into it.
    std::vector<std::string> background_ids;
    std::set<std::string> base_leg_ids;
    for (const auto& work : merged_base.background_works) {
        if (!candidate_ids.count(work.work_id)) {
            if (explicit_background_ids.count(work.work_id)) {
                background_ids.push_back(work.work_id);
            }
            for (const auto& leg : work.legs) base_leg_ids.insert(leg.leg_id);
        }
    }
    const NativeProjectionContext::ReferenceRunResult* reference_run =
        nullptr;
    std::string ref_status;
    if (options.include_background_reference && !background_ids.empty()) {
        uint64_t reference_key = 0;
        {
            std::string key_material = "reference-v1|"
                + overlay_wire.dump()
                + "|" + std::to_string(effective_window_limit)
                + "|" + std::to_string(options.projection_event_budget)
                + "|" + (options.has_horizon
                             ? std::to_string(options.horizon_ns) : "-")
                + "|" + (options.drain_background ? "d1" : "d0")
                + "|" + (options.analytic_segments ? "a1" : "a0");
            uint64_t hash = 14695981039346656037ULL;
            for (char c : key_material) {
                hash ^= static_cast<uint64_t>(static_cast<unsigned char>(c));
                hash *= 1099511628211ULL;
            }
            reference_key = hash;
        }
        reference_run = context.lookup_reference_run(reference_key);
        if (reference_run == nullptr) {
            // Miss: consume merged_base for the reference build (sparing a
            // third deep copy); the candidate world re-applies the overlay.
            NativeProjectionInput reference_input = std::move(merged_base);
            reference_input.background_works.erase(
                std::remove_if(reference_input.background_works.begin(),
                               reference_input.background_works.end(),
                               [&](const NativeProjectionWorkSpec& work) {
                                   return candidate_ids.count(work.work_id) != 0;
                               }),
                reference_input.background_works.end());
            reference_input.explicit_background_work_ids.erase(
                std::remove_if(
                    reference_input.explicit_background_work_ids.begin(),
                    reference_input.explicit_background_work_ids.end(),
                    [&](const std::string& id) {
                        return candidate_ids.count(id) != 0;
                    }),
                reference_input.explicit_background_work_ids.end());
            reference_input.barriers.erase(std::remove_if(reference_input.barriers.begin(), reference_input.barriers.end(),
                [&](const NativeProjectionBarrier& barrier) {
                    return std::any_of(barrier.signallers.begin(), barrier.signallers.end(),
                        [&](const std::string& id) { return !base_leg_ids.count(id); });
                }), reference_input.barriers.end());
            NativeProjectionContext::ReferenceRunResult recorded;
            {
                NativeProjectionWorld reference(
                    reference_input, context.contract_ptr(),
                    effective_window_limit, options.projection_event_budget);
                ref_status = reference.run(options.has_horizon, options.horizon_ns,
                    options.projection_event_budget, options.drain_background,
                    options.analytic_segments);
                for (const auto& work : reference.input().background_works) {
                    recorded.completion_by_work.emplace(
                        work.work_id, reference.work_completion(work.work_id));
                }
            }
            recorded.status = ref_status;
            reference_run = context.store_reference_run(
                reference_key, std::move(recorded));
            merged_base = apply_projection_overlay(context.input(), overlay);
        } else {
            ref_status = reference_run->status;
        }
    }
    if (!candidates.empty()) merged_base.focus_work_id = candidates.front().work_id;
    for (const auto& work : candidates) merged_base.background_works.push_back(work);
    NativeProjectionWorld world(
        merged_base, context.contract_ptr(), effective_window_limit,
        options.projection_event_budget);
    const std::string focus_id = merged_base.focus_work_id;
    uint64_t closure_work_count = closure.works.size();
    uint64_t closure_resource_count = closure.resources.size();

    auto result_base = [&]() {
        Json out{{"status", kStatusUnavailable},
                 {"predicted_service_done_ns", nullptr},
                 {"predicted_settle_done_ns", nullptr},
                 {"reason", ""}, {"dependency_attribution", Json::array()},
                 {"focus_bytes", 0}, {"bytes_served_total", 0},
                 {"event_count", 0}, {"resource_count", world.resource_count()},
                 {"work_count", world.work_count()},
                 {"closure_resource_count", closure_resource_count},
                 {"closure_work_count", closure_work_count},
                 {"approximations", Json::array()}, {"fallback", nullptr},
                 {"background_comparisons", Json::array()},
                 {"reference_cache_hits", context.reference_cache_hits()},
                 {"reference_cache_misses", context.reference_cache_misses()}};
        out.update(world.window_quality_fields(kStatusUnavailable));
        return out;
    };
    if (closure_resource_count > options.closure_resource_budget) {
        Json out = result_base();
        out["reason"] = "closure_resource_budget_exceeded";
        out["fallback"] = "closure_resource_budget_exceeded";
        out.update(world.window_quality_fields(kStatusUnavailable));
        out["unsupported_reason"] = "closure_resource_budget_exceeded";
        return out;
    }

    const std::string status = world.run(options.has_horizon, options.horizon_ns,
        options.projection_event_budget, options.drain_background,
        options.analytic_segments);
    Json out = result_base();
    out["status"] = status;
    out["event_count"] = world.events_processed();
    out["bytes_served_total"] = static_cast<uint64_t>(world.bytes_served_total());
    out["resource_count"] = world.resource_count();
    out["work_count"] = world.work_count();
    out["dependency_attribution"] = world.attribution(focus_id);
    out.update(world.window_quality_fields(status));
    if (status == kStatusDegraded) {
        // The legacy result contract never exposes costs from a truncated
        // candidate, even though the mutable internal world has progressed.
        out["bytes_served_total"] = 0;
        out["reason"] = "event_budget_exceeded";
        out["fallback"] = "event_budget_exceeded";
        return out;
    }

    out["predicted_settle_done_ns"] = world.work_completion(focus_id)
        ? Json(*world.work_completion(focus_id)) : Json(nullptr);
    out["predicted_service_done_ns"] = world.service_completion(focus_id)
        ? Json(*world.service_completion(focus_id)) : Json(nullptr);
    if (status == kStatusWait) out["reason"] = "horizon reached with events still in flight (legal wait)";
    else if (status == kStatusBlocked) out["reason"] = "no progress path (zero-rate stall or held gates)";
    else out["reason"] = "";
    if (!world.work_completion(focus_id)) out["dependency_attribution"] = world.attribution(focus_id);
    else out["dependency_attribution"] = Json::array();

    uint64_t focus_bytes = 0;
    const auto add_focus_bytes = [&](uint64_t amount) {
        if (amount > std::numeric_limits<uint64_t>::max() - focus_bytes) {
            fail("focus_bytes", "sum overflows uint64");
        }
        focus_bytes += amount;
    };
    for (const auto& work : world.input().background_works) {
        if (work.work_id == focus_id) {
            for (const auto& leg : work.legs) {
                add_focus_bytes(leg.bytes_amount);
            }
            for (const auto& descriptor : work.pipeline_descriptors) {
                for (const auto& chain : descriptor->projection().chains) {
                    add_focus_bytes(projection_fill_bytes_in_chain(
                        *descriptor, chain));
                    if (chain.consume.has_hbm_share) {
                        add_focus_bytes(projection_share_units_in_chain(
                            chain.consume.hbm_share, chain.tile_count,
                            "focus_bytes.hbm_share"));
                    }
                }
            }
        }
    }
    out["focus_bytes"] = focus_bytes;

    // The reference run already executed (or was memoized) ahead of the
    // candidate world above; `background_ids` and `base_leg_ids` were
    // captured from the pre-candidate merged input there.
    if (options.include_background_reference && !background_ids.empty()
            && reference_run != nullptr) {
        for (const auto& work_id : background_ids) {
            const auto found =
                reference_run->completion_by_work.find(work_id);
            const std::optional<uint64_t> without_candidate =
                found == reference_run->completion_by_work.end()
                    ? std::nullopt : found->second;
            out["background_comparisons"].push_back(background_comparison(
                work_id, without_candidate,
                world.work_completion(work_id)));
        }
        if (ref_status != kStatusOk) out["approximations"].push_back("background_reference_" + ref_status);
    }
    std::vector<std::string> unfinished;
    for (const auto& work_id : background_ids) {
        if (!world.work_completion(work_id)) unfinished.push_back(work_id);
    }
    if (!unfinished.empty()) {
        std::sort(unfinished.begin(), unfinished.end());
        std::string joined;
        for (size_t i = 0; i < unfinished.size(); ++i) {
            if (i) joined += ",";
            joined += unfinished[i];
        }
        out["approximations"].push_back("background_drain_truncated:" + joined);
    }
    return out;
}

struct NativeProjectionWorldSession::Impl final {
    explicit Impl(std::shared_ptr<const NativeProjectionContext> value)
        : context(std::move(value)), input(context->input()),
          world(std::make_unique<NativeProjectionWorld>(
              input, context->contract_ptr(), 1, 0)) {}
    std::shared_ptr<const NativeProjectionContext> context;
    NativeProjectionInput input;
    std::unique_ptr<NativeProjectionWorld> world;
    uint64_t event_budget = 0;
    bool conservation_check = true;

    void reset(const Json& request) {
        if (request.contains("event_budget")) event_budget = uint_value(request.at("event_budget"), "reset.event_budget");
        else event_budget = 0;
        if (request.contains("conservation_check")) {
            if (!request.at("conservation_check").is_boolean()) fail("reset.conservation_check", "expected boolean");
            conservation_check = request.at("conservation_check").get<bool>();
        } else conservation_check = true;
        world = std::make_unique<NativeProjectionWorld>(
            input, context->contract_ptr(), 1, event_budget);
        world->set_conservation_check(conservation_check);
    }
};

NativeProjectionWorldSession::NativeProjectionWorldSession(
    std::shared_ptr<const NativeProjectionContext> context)
    : impl_(std::make_unique<Impl>(std::move(context))) {
    if (!impl_->context) fail("world_session", "context is null");
}
NativeProjectionWorldSession::~NativeProjectionWorldSession() = default;
NativeProjectionWorldSession::NativeProjectionWorldSession(NativeProjectionWorldSession&&) noexcept = default;
NativeProjectionWorldSession& NativeProjectionWorldSession::operator=(NativeProjectionWorldSession&&) noexcept = default;

nlohmann::json NativeProjectionWorldSession::operation(const nlohmann::json& request) {
    if (!request.is_object() || !request.contains("operation") || !request.at("operation").is_string()) {
        fail("world_operation", "request needs string operation");
    }
    const std::string op = request.at("operation").get<std::string>();
    if (op == "reset") {
        object(request, "world_operation", {"operation"}, {"event_budget", "conservation_check"});
        impl_->reset(request);
        return impl_->world->inspect();
    }
    if (op == "inspect") {
        object(request, "world_operation", {"operation"});
        return impl_->world->inspect();
    }
    if (op == "step") {
        object(request, "world_operation", {"operation"});
        const bool progressed = impl_->world->advance_one();
        Json out = impl_->world->inspect();
        out["progressed"] = progressed;
        return out;
    }
    if (op == "run") {
        object(request, "world_operation", {"operation"},
               {"horizon_ns", "event_budget", "analytic_segments", "drain_background"});
        bool has_horizon = false;
        double horizon = 0.0;
        if (request.contains("horizon_ns") && !request.at("horizon_ns").is_null()) {
            horizon = number_value(request.at("horizon_ns"), "run.horizon_ns");
            if (horizon < 0.0) fail("run.horizon_ns", "must be nonnegative");
            has_horizon = true;
        }
        uint64_t budget = impl_->event_budget;
        if (request.contains("event_budget") && !request.at("event_budget").is_null()) budget = uint_value(request.at("event_budget"), "run.event_budget");
        bool analytic = true, drain = true;
        if (request.contains("analytic_segments")) {
            if (!request.at("analytic_segments").is_boolean()) fail("run.analytic_segments", "expected boolean");
            analytic = request.at("analytic_segments").get<bool>();
        }
        if (request.contains("drain_background")) {
            if (!request.at("drain_background").is_boolean()) fail("run.drain_background", "expected boolean");
            drain = request.at("drain_background").get<bool>();
        }
        const auto status = impl_->world->run(has_horizon, horizon, budget, drain, analytic);
        Json out = impl_->world->inspect();
        out["status"] = status;
        return out;
    }
    if (op == "integrate") {
        object(request, "world_operation", {"operation", "target_ns"});
        Json out = impl_->world->integrate_controlled(number_value(request.at("target_ns"), "integrate.target_ns"));
        out.update(impl_->world->inspect());
        return out;
    }
    if (op == "assert_conservation") {
        object(request, "world_operation", {"operation", "pre_remaining", "served_total", "rates", "interval_start_ns"});
        impl_->world->assert_conservation_controlled(
            number_value(request.at("pre_remaining"), "assert_conservation.pre_remaining"),
            number_value(request.at("served_total"), "assert_conservation.served_total"),
            request.at("rates"),
            number_value(request.at("interval_start_ns"), "assert_conservation.interval_start_ns"),
            impl_->world->now_ns());
        return Json{{"ok", true}};
    }
    if (op == "set_clock") {
        object(request, "world_operation", {"operation", "now_ns"});
        impl_->world->set_clock(number_value(request.at("now_ns"), "set_clock.now_ns"));
        return impl_->world->inspect();
    }
    fail("world_operation", "unknown operation '" + op + "'");
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
