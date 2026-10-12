/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/Sys.hh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <utility>

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/BaseStream.hh"
#include "astra-sim/system/CollectivePlan.hh"
#include "astra-sim/system/DataSet.hh"
#include "astra-sim/system/MemBus.hh"
#include "astra-sim/system/QueueLevels.hh"
#include "astra-sim/system/RendezvousRecvData.hh"
#include "astra-sim/system/RendezvousSendData.hh"
#include "astra-sim/system/SendPacketEventHandlerData.hh"
#include "astra-sim/system/StreamBaseline.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include "astra-sim/system/astraccl/CollectiveImpl.hh"
#include "astra-sim/system/astraccl/custom_collectives/CustomAlgorithm.hh"
#include "astra-sim/system/astraccl/native_collectives/collective_algorithm/AllToAll.hh"
#include "astra-sim/system/astraccl/native_collectives/collective_algorithm/DoubleBinaryTreeAllReduce.hh"
#include "astra-sim/system/astraccl/native_collectives/collective_algorithm/HalvingDoubling.hh"
#include "astra-sim/system/astraccl/native_collectives/collective_algorithm/Ring.hh"
#include "astra-sim/system/scheduling/OfflineGreedy.hh"
#include "astra-sim/system/astraccl/native_collectives/logical_topology/BasicLogicalTopology.hh"
#include "astra-sim/system/astraccl/native_collectives/logical_topology/GeneralComplexTopology.hh"
#include "DieModel.hh"
#include <json/json.hpp>

using namespace std;
using namespace Chakra;
using json = nlohmann::json;

namespace AstraSim {
uint8_t* Sys::dummy_data = new uint8_t[2];
vector<Sys*> Sys::all_sys;

namespace {

void cancel_call_events_alarm(void* const arg) {
    delete static_cast<BasicEventHandlerData*>(arg);
}

// Parse a system.json flag accepted as boolean or integer (0/1).  A bare
// `j["key"] != 0` must not be used: nlohmann::json never equates a boolean
// with a number, so a flag written as false would satisfy `!= 0` and
// fail-open.  Any other JSON type is a configuration error.
bool parse_bool_or_int_flag(const json& value, const string& key) {
    if (value.is_boolean()) {
        return value.get<bool>();
    }
    if (value.is_number_integer() || value.is_number_unsigned()) {
        return value.get<int64_t>() != 0;
    }
    Sys::sys_panic(key + " must be boolean or integer");
    return false;  // unreachable; sys_panic exits
}

// R7-B fix (2026-09-29): strict readers for the numeric system.json keys.
// The implicit nlohmann conversions had three failure shapes: a wrong-typed
// value escaped as an uncaught json type_error through the Sys ctor chain
// (terminate); a negative integer wrapped inside unsigned members
// (local-mem-latency: -1 -> 1.8e19 -> nodes that never complete) or produced
// negative event deltas that abort the event queue far away from the config
// key; a fractional value silently truncated. Every numeric key now parses
// through these readers, which fail closed naming the offending key.
double parse_non_negative_number(const json& value, const string& key) {
    if (!value.is_number()) {
        Sys::sys_panic("system config key '" + key + "' must be a number");
    }
    const double v = value.get<double>();
    if (!std::isfinite(v) || v < 0.0) {
        Sys::sys_panic("system config key '" + key +
                       "' must be a finite non-negative number, got " +
                       value.dump());
    }
    return v;
}

int parse_non_negative_int(const json& value, const string& key,
                           int minimum) {
    int64_t v = 0;
    if (value.is_number_unsigned()) {
        const uint64_t u = value.get<uint64_t>();
        if (u > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            Sys::sys_panic("system config key '" + key +
                           "' exceeds the 64-bit integer range");
        }
        v = static_cast<int64_t>(u);
    } else if (value.is_number_integer()) {
        v = value.get<int64_t>();
    } else {
        Sys::sys_panic("system config key '" + key +
                       "' must be an integer, got " + value.dump());
    }
    if (v < minimum) {
        Sys::sys_panic("system config key '" + key + "' must be >= " +
                       std::to_string(minimum) + ", got " +
                       std::to_string(v));
    }
    if (v > std::numeric_limits<int>::max()) {
        Sys::sys_panic("system config key '" + key +
                       "' exceeds the int range");
    }
    return static_cast<int>(v);
}

uint64_t parse_non_negative_u64(const json& value, const string& key) {
    if (value.is_number_unsigned()) {
        return value.get<uint64_t>();
    }
    if (value.is_number_integer()) {
        const int64_t v = value.get<int64_t>();
        if (v < 0) {
            Sys::sys_panic("system config key '" + key +
                           "' must be non-negative, got " + std::to_string(v));
        }
        return static_cast<uint64_t>(v);
    }
    Sys::sys_panic("system config key '" + key +
                   "' must be an integer, got " + value.dump());
    return 0;  // unreachable; sys_panic exits
}

// M2 fix (2026-09-29): the rendezvous handshake is a fixed 8192-byte control
// transfer that used to point straight at the caller's message buffer, so any
// backend that really delivers into a receive buffer would overflow a message
// smaller than 8192 bytes. Neither shipped backend dereferences handshake
// bytes (control-only transfer, single-threaded event loop), so one shared
// buffer carries the transfer instead of the user payload. Do not resize the
// user buffer to 8192: the handshake size is part of the modeled timing.
uint8_t g_rendezvous_handshake_buffer[8192];

// Process-level die plan-cache registry (stage E2): one shared LRU PlanCache
// per resolved hardware summary. compile() is a pure function and a DiePlan
// is immutable, so every Sys running the same die hardware shares its
// compiled plans instead of recompiling identical plan_cache_keys per rank
// (design doc §8.5: "计划可以缓存"). The simulation is single-threaded (one
// event loop drives all Sys), so the registry needs no lock; if a
// multi-threaded runtime is ever introduced, guard this function and the
// shared DieModel::PlanCache with a mutex.
std::shared_ptr<DieModel::PlanCache> shared_die_plan_cache(
    const DieModel::Hardware& hw) {
    static std::map<std::string, std::shared_ptr<DieModel::PlanCache>>
        registry;
    const std::string key = DieModel::plan_cache_hardware_key(hw);
    auto it = registry.find(key);
    if (it == registry.end()) {
        it = registry
                 .emplace(key, std::make_shared<DieModel::PlanCache>(4096))
                 .first;
    }
    return it->second;
}

}  // namespace

// SchedulerUnit --------------------------------------------------------------
Sys::SchedulerUnit::SchedulerUnit(Sys* sys,
                                  vector<int> queues,
                                  int max_running_streams,
                                  int ready_list_threshold,
                                  int queue_threshold) {
    this->sys = sys;
    this->max_running_streams = max_running_streams;
    this->ready_list_threshold = ready_list_threshold;
    this->queue_threshold = queue_threshold;
    this->total_active_chunks_per_dimension.resize(queues.size(), 0);

    int base = 0;
    int dimension = 0;
    for (auto q : queues) {
        for (int i = 0; i < q; i++) {
            this->running_streams[base] = 0;
            this->queue_id_to_dimension[base] = dimension;
            base++;
        }
        dimension++;
        UsageTracker u(2);
        usage.push_back(u);
    }
}

void Sys::SchedulerUnit::notify_stream_added(int vnet) {
    if (sys->id == 0 &&
        ++total_active_chunks_per_dimension[queue_id_to_dimension[vnet]] == 1) {
        usage[queue_id_to_dimension[vnet]].increase_usage();
    }
    // Scheduling cursor: purely local to this call (re-seeded from begin()
    // every time, never read across calls -- the former member only stored
    // a default-constructed iterator between invocations).
    list<BaseStream*>::iterator pointer = sys->active_Streams[vnet].begin();
    advance(pointer, running_streams[vnet]);
    while (pointer != sys->active_Streams[vnet].end() &&
           running_streams[vnet] < queue_threshold) {
        (*pointer)->init();
        running_streams[vnet]++;
        advance(pointer, 1);
    }
}

void Sys::SchedulerUnit::notify_stream_added_into_ready_list() {
    if (this->sys->first_phase_streams < ready_list_threshold &&
        this->sys->total_running_streams < max_running_streams) {
        int max = ready_list_threshold - sys->first_phase_streams;
        if (max > max_running_streams - this->sys->total_running_streams) {
            max = max_running_streams - this->sys->total_running_streams;
        }
        sys->schedule(max);
    }
    return;
}

void Sys::SchedulerUnit::notify_stream_removed(int vnet) {
    if (sys->id == 0 &&
        --total_active_chunks_per_dimension[queue_id_to_dimension[vnet]] == 0) {
        usage[queue_id_to_dimension[vnet]].decrease_usage();
    }
    running_streams[vnet]--;

    if (this->sys->first_phase_streams < ready_list_threshold &&
        this->sys->total_running_streams < max_running_streams) {
        int max = ready_list_threshold - sys->first_phase_streams;
        if (max > max_running_streams - this->sys->total_running_streams) {
            max = max_running_streams - this->sys->total_running_streams;
        }
        sys->schedule(max);
    }
    // Same local scheduling cursor as notify_stream_added.
    list<BaseStream*>::iterator pointer = sys->active_Streams[vnet].begin();
    advance(pointer, running_streams[vnet]);
    while (pointer != sys->active_Streams[vnet].end() &&
           running_streams[vnet] < queue_threshold) {
        (*pointer)->init();
        running_streams[vnet]++;
        advance(pointer, 1);
    }
}
//-----------------------------------------------------------------------------

Sys::Sys(int id,
         string workload_configuration,
         string comm_group_configuration,
         string system_configuration,
         AstraNetworkAPI* comm_NI,
         vector<int> physical_dims,
         vector<int> queues_per_dim,
         double injection_scale,
         double /*comm_scale -- member removed (write-only); parameter kept
                    for the unchanged call sites in the frontend/tests */,
         bool rendezvous_enabled,
         ExecutionDriven::ExecutionMode execution_mode,
         std::shared_ptr<ExecutionDriven::GraphSource> graph_source) {
    this->execution_mode_ = execution_mode;

    if ((id + 1) > this->all_sys.size()) {
        this->all_sys.resize(id + 1);
    }
    this->all_sys[id] = this;

    this->id = id;

    this->workload = nullptr;

    this->roofline_enabled = false;
    this->peak_perf = 0;
    this->roofline = nullptr;
    this->compute_cost_model = ComputeCostModel::LegacyDefault;
    this->die_runtime = nullptr;
    this->die_model_config_path = "";

    this->local_mem_bw = 0;
    this->local_mem_latency = 0;
    this->hbm_bandwidth_contention = false;

    this->memBus = nullptr;
    this->inp_L = 0;
    this->inp_o = 0;
    this->inp_g = 0;
    this->inp_G = 0;
    this->model_shared_bus = 0;
    this->injection_scale = injection_scale;

    this->comm_NI = comm_NI;
    this->rendezvous_enabled = rendezvous_enabled;

    this->scheduler_unit = nullptr;
    this->vLevels = nullptr;
    this->offline_greedy = nullptr;
    this->intra_dimension_scheduling = IntraDimensionScheduling::FIFO;
    this->inter_dimension_scheduling = InterDimensionScheduling::Ascending;
    this->scheduling_policy = SchedulingPolicy::FIFO;
    this->round_robin_inter_dimension_scheduler = 0;
    this->active_chunks_per_dimension = 1;
    this->priority_counter = 0;
    this->preferred_dataset_splits = 1;

    this->collectiveOptimization = CollectiveOptimization::Baseline;
    this->last_scheduled_collective = 0;

    this->first_phase_streams = 0;
    this->total_running_streams = 0;

    this->communication_delay = 10;
    this->local_reduction_delay = 1;

    collective_impl_lookup = new CollectiveImplLookup(id);

    // initialize_sys never returns false: every failure path exits inside
    // (missing file logs critical and exits(1)).
    initialize_sys(system_configuration);

    // scheduler
    this->physical_dims = physical_dims;
    this->queues_per_dim = queues_per_dim;
    int element = 0;
    this->total_nodes = 1;
    for (uint64_t current_dim = 0; current_dim < queues_per_dim.size();
         current_dim++) {
        if (physical_dims[current_dim] >= 1) {
            this->total_nodes *= physical_dims[current_dim];
        }
        for (int j = 0; j < queues_per_dim[current_dim]; j++) {
            list<BaseStream*> temp;
            active_Streams[element] = temp;
            element++;
        }
    }

    // Fail-loud before the divide below: queues_per_dim[0] == 0 turns this
    // into a double division by zero whose (int)ceil(inf) conversion is UB
    // (typically INT_MIN, which silently disables all stream init). Legal
    // configurations always carry >= 1 queue in the first dimension; the
    // zero-queue-dimension panic in proceed() would otherwise only fire
    // much later, after this ctor-time UB.
    if (queues_per_dim.empty() || queues_per_dim[0] <= 0) {
        sys_panic("Sys ctor: queues_per_dim is empty or its first dimension "
                  "has zero queues -- cannot derive concurrent_streams "
                  "(fail-closed)");
    }
    // Local scheduler-tuning constants: consumed only by the SchedulerUnit
    // below, never stored on Sys.
    const int concurrent_streams =
        (int)ceil(((double)active_chunks_per_dimension) / queues_per_dim[0]);
    const int active_first_phase = 100000000;
    const int max_running = 100000000;

    scheduler_unit = new SchedulerUnit(this, queues_per_dim, max_running,
                                       active_first_phase, concurrent_streams);

    vLevels = new QueueLevels(queues_per_dim, 0, comm_NI->get_backend_type());

    // collective communication
    this->num_streams = 0;

    logical_topologies["AllReduce"] = new GeneralComplexTopology(
        id, physical_dims, collective_impl_lookup->get_collective_impl(ComType::All_Reduce, 0, BypassRule::BYPASS_ALL_CUSTOM));
    logical_topologies["ReduceScatter"] = new GeneralComplexTopology(
        id, physical_dims, collective_impl_lookup->get_collective_impl(ComType::Reduce_Scatter, 0, BypassRule::BYPASS_ALL_CUSTOM));
    logical_topologies["AllGather"] = new GeneralComplexTopology(
        id, physical_dims, collective_impl_lookup->get_collective_impl(ComType::All_Gather, 0, BypassRule::BYPASS_ALL_CUSTOM));
    logical_topologies["AllToAll"] = new GeneralComplexTopology(
        id, physical_dims, collective_impl_lookup->get_collective_impl(ComType::All_to_All, 0, BypassRule::BYPASS_ALL_CUSTOM));

    memBus = new MemBus("NPU", "MA", this, inp_L, inp_o, inp_g, inp_G,
                        model_shared_bus, communication_delay, true);

    // step-1-2 execution-mode factory: online mode injects the dynamic
    // GraphSource at Sys creation (never constructs the ETFeeder, never
    // requires .et files); the static path stays byte-for-byte unchanged.
    if (execution_mode_ == ExecutionDriven::ExecutionMode::Online) {
        workload =
            new Workload(this, workload_configuration,
                         comm_group_configuration, execution_mode_,
                         graph_source);
    } else {
        workload =
            new Workload(this, workload_configuration,
                         comm_group_configuration);
    }

    if (inter_dimension_scheduling == InterDimensionScheduling::OfflineGreedy) {
        offline_greedy = new OfflineGreedy(this);
    }
}

Sys::~Sys() {
    if (roofline_enabled) {
        delete this->roofline;
    }

    all_sys[id] = nullptr;

    for (auto lt : logical_topologies) {
        delete lt.second;
    }

    logical_topologies.clear();

    if (scheduler_unit != nullptr) {
        delete scheduler_unit;
    }

    if (vLevels != nullptr) {
        delete vLevels;
    }

    if (memBus != nullptr) {
        delete memBus;
    }

    if (workload != nullptr) {
        delete workload;
    }

    if (offline_greedy != nullptr) {
        delete offline_greedy;
    }

    if (collective_impl_lookup != nullptr) {
        delete collective_impl_lookup;
    }

    bool shouldExit = true;
    for (auto& a : all_sys) {
        if (a != nullptr) {
            shouldExit = false;
            break;
        }
    }

    if (shouldExit) {
        auto logger = LoggerFactory::get_logger("system");
        logger->warn("Exiting");
    }
}

bool Sys::initialize_sys(string name) {
    ifstream inFile;
    inFile.open(name);
    if (!inFile) {
        if (id == 0) {
            LoggerFactory::get_logger("system")->critical(
                "Unable to open file: {}", name);
        }
        exit(1);
    }

    json j;
    // R5-Q1 fix (2026-09-29): fail-closed system.json loading, same shape as
    // the comm-group loader (Workload.cc S6/R2-3). The bare `inFile >> j`
    // had two failure modes: a malformed file threw an uncaught nlohmann
    // parse_error through the Sys ctor chain (terminate), and a syntactically
    // valid NON-OBJECT document made every contains() below false, so the
    // whole config was silently dropped and the run proceeded on defaults
    // (roofline off, peak_perf 0 -- a silently changed simulation scale).
    try {
            inFile >> j;
        } catch (const std::exception& e) {
            sys_panic("failed to parse system config file " + name + ": " +
                      e.what());
        }
        if (!j.is_object()) {
            sys_panic("system config file " + name +
                      " must contain a JSON object");
        }

        // R5-Q1: one fail-closed wrapper for the whole key-walk. A wrong-typed
        // value for any key (e.g. a string where a number belongs) escapes as a
        // json type_error; route it through sys_panic with the file name instead
        // of letting it terminate through the unguarded `new Sys` call chain.
        // The strict numeric readers above exit directly and never throw.
        // Stage-B backend-resolution state: the compute-cost-model keys are
        // parsed inside the walk below; the conflict matrix and the die
        // hardware load (initialize_die_backend below) run after the walk.
        bool explicit_compute_cost_model_present = false;
        ComputeCostModel explicit_compute_cost_model =
            ComputeCostModel::LegacyDefault;
        bool die_model_profile_present = false;
        string die_model_profile;
        bool die_model_config_explicit_present = false;
        string die_model_config_path;
        const bool peak_perf_key_present = j.contains("peak-perf");
        const bool local_mem_bw_key_present = j.contains("local-mem-bw");
        const bool hbm_contention_key_present =
            j.contains("hbm-bandwidth-contention");
        try {
        if (j.contains("scheduling-policy")) {
            if (!j["scheduling-policy"].is_string()) {
                Sys::sys_panic(
                    "system config key 'scheduling-policy' must be a "
                    "string");
            }
            string inp_scheduling_policy = j["scheduling-policy"];
            if (inp_scheduling_policy == "LIFO") {
                this->scheduling_policy = SchedulingPolicy::LIFO;
            } else if (inp_scheduling_policy == "FIFO") {
                this->scheduling_policy = SchedulingPolicy::FIFO;
            } else if (inp_scheduling_policy == "EXPLICIT") {
                this->scheduling_policy = SchedulingPolicy::EXPLICIT;
            } else {
                sys_panic("unknown value for scheduling policy in sys input file");
            }
        }
        if (j.contains("inter-dimension-scheduling")) {
            if (!j["inter-dimension-scheduling"].is_string()) {
                Sys::sys_panic("system config key "
                               "'inter-dimension-scheduling' must be a "
                               "string");
            }
            string inp_inter_dimension_scheduling =
                j["inter-dimension-scheduling"];
            if (inp_inter_dimension_scheduling == "Ascending") {
                inter_dimension_scheduling = InterDimensionScheduling::Ascending;
            } else if (inp_inter_dimension_scheduling == "RoundRobin") {
                inter_dimension_scheduling = InterDimensionScheduling::RoundRobin;
            } else if (inp_inter_dimension_scheduling == "OfflineGreedy") {
                inter_dimension_scheduling =
                    InterDimensionScheduling::OfflineGreedy;
            } else {
                sys_panic("unknown value for inter-dimension-scheduling in sys "
                          "input file");
            }
        }
        if (j.contains("intra-dimension-scheduling")) {
            if (!j["intra-dimension-scheduling"].is_string()) {
                Sys::sys_panic("system config key "
                               "'intra-dimension-scheduling' must be a "
                               "string");
            }
            string inp_intra_dimension_scheduling =
                j["intra-dimension-scheduling"];
            if (inp_intra_dimension_scheduling == "FIFO") {
                intra_dimension_scheduling = IntraDimensionScheduling::FIFO;
            } else if (inp_intra_dimension_scheduling == "RG") {
                intra_dimension_scheduling = IntraDimensionScheduling::RG;
            } else if (inp_intra_dimension_scheduling == "SmallestFirst") {
                intra_dimension_scheduling =
                    IntraDimensionScheduling::SmallestFirst;
            } else if (inp_intra_dimension_scheduling ==
                       "LessRemainingPhaseFirst") {
                intra_dimension_scheduling =
                    IntraDimensionScheduling::LessRemainingPhaseFirst;
            } else {
                sys_panic("unknown value for intra-dimension-scheduling in sys "
                          "input file");
            }
        }
        if (j.contains("collective-optimization")) {
            if (!j["collective-optimization"].is_string()) {
                Sys::sys_panic("system config key 'collective-optimization' "
                               "must be a string");
            }
            string inp_collective_optimization = j["collective-optimization"];
            if (inp_collective_optimization == "baseline") {
                collectiveOptimization = CollectiveOptimization::Baseline;
            } else if (inp_collective_optimization == "localBWAware") {
                collectiveOptimization = CollectiveOptimization::LocalBWAware;
            } else {
                sys_panic(
                    "unknown value for collective optimization in sys input file");
            }
        }
        if (j.contains("local-reduction-delay")) {
            local_reduction_delay = parse_non_negative_int(
                j["local-reduction-delay"], "local-reduction-delay", 0);
        }
        // M7/R8-X1 fix: active-chunks-per-dimension <= 0 drives
        // queue_threshold <= 0, so notify_stream_added never inits any stream
        // (the only two init() call sites live inside its threshold loop) and
        // the run stalls with a rootless lost-wakeup. The value must be >= 1.
        if (j.contains("active-chunks-per-dimension")) {
            active_chunks_per_dimension = parse_non_negative_int(
                j["active-chunks-per-dimension"], "active-chunks-per-dimension",
                1);
        }
        if (j.contains("L")) {
            inp_L = parse_non_negative_number(j["L"], "L");
        }
        if (j.contains("o")) {
            inp_o = parse_non_negative_number(j["o"], "o");
        }
        if (j.contains("g")) {
            inp_g = parse_non_negative_number(j["g"], "g");
        }
        if (j.contains("G")) {
            inp_G = parse_non_negative_number(j["G"], "G");
        }
        if (j.contains("endpoint-delay")) {
            communication_delay = parse_non_negative_int(j["endpoint-delay"],
                                                         "endpoint-delay", 0);
            // Bound the scaled product before the double->int write-back
            // (2026-09-30): a legal but extreme --injection-scale makes the
            // conversion UB. Truncation semantics for fractional scales are
            // kept unchanged (cross-repo timing comparability).
            const double scaled =
                static_cast<double>(communication_delay) * injection_scale;
            if (!std::isfinite(scaled) ||
                scaled > static_cast<double>(
                             std::numeric_limits<int>::max())) {
                Sys::sys_panic(
                    "system config: endpoint-delay x injection-scale exceeds "
                    "the int delay range (" + std::to_string(scaled) + ")");
            }
            communication_delay = static_cast<int>(scaled);
        }
        if (j.contains("model-shared-bus")) {
            // Same parse contract as the other boolean flags (see
            // parse_bool_or_int_flag): a bare int cast would throw a nlohmann
            // type error for a JSON `true`/`false` spelling.
            model_shared_bus = parse_bool_or_int_flag(j["model-shared-bus"],
                                                      "model-shared-bus");
        } else {
            model_shared_bus = false;
        }
        if (model_shared_bus) {
            // Fail-closed coefficient gate (2026-09-30): L/o/g/G all
            // default to 0 when absent, so enabling the shared-bus model
            // without spelling out its coefficients silently collapsed the
            // LogGP timing to instantaneous transfers (a dormant path --
            // no shipped config sets model-shared-bus, which is exactly why
            // the degenerate shape had no test coverage).
            for (const char* key : {"L", "o", "g", "G"}) {
                if (!j.contains(key)) {
                    Sys::sys_panic(std::string("system config: ") +
                                   "model-shared-bus is enabled but " +
                                   "coefficient '" + key +
                                   "' is missing (the silent default of 0 "
                                   "collapses the LogGP transfer model)");
                }
            }
        }
        // M6/R6-T1 fix: preferred-dataset-splits <= 0 used to be accepted here
        // and then silently bypassed the total_nodes chunk-size floor inside
        // determine_chunk_size, shrinking below-node collectives to zero-byte
        // messages that exit(-1) in the fluid backend. The key must be >= 1
        // (missing keeps the default of 1, which never reaches this branch).
        if (j.contains("preferred-dataset-splits")) {
            preferred_dataset_splits = parse_non_negative_int(
                j["preferred-dataset-splits"], "preferred-dataset-splits", 1);
        }
        if (j.contains("peak-perf")) {
            // R7-B discipline (2026-09-30): these two keys were the last
            // numeric configs still read through nlohmann's implicit
            // conversion -- a negative value flowed silently into the
            // Roofline/HBM models as a negative rate. Parse strictly like
            // every other numeric key.
            peak_perf = parse_non_negative_number(j["peak-perf"],
                                                  "peak-perf");
            peak_perf = peak_perf * 1000000000000;  // TFLOPS
            // (2026-10-01) post-scale finiteness: the check inside the
            // parser ran on the PRE-scale value, so e.g. 1e300 passed and
            // became inf here -- an infinite peak silently turns every
            // Roofline elapsed into 0 instead of failing closed.
            if (!std::isfinite(peak_perf)) {
                Sys::sys_panic("system config: peak-perf x 1e12 is not "
                               "finite");
            }
        }
        // Unit note: the hardware-template source key for this value is
        // historically named "bandwidth-gbps", but its value is GB/s
        // (gigaBYTES per second) across this repo -- the generator, network.yml,
        // the Python scheduler and MetricCollector's
        // local_hbm_bw_bytes_per_second all treat it as byte-oriented, and the
        // x1e9 below converts GB/s to bytes/s. The name is legacy; do not read
        // it as Gbit/s (that would mis-scale cross-repo comparisons by 8x).
        if (j.contains("local-mem-bw")) {
            local_mem_bw = parse_non_negative_number(j["local-mem-bw"],
                                                    "local-mem-bw");
            local_mem_bw = local_mem_bw * 1000000000;  // GB/sec -> bytes/sec
            // (2026-10-01) post-scale finiteness, same reason as peak-perf
            // (an inf bandwidth silently zeroes every HBM elapsed).
            if (!std::isfinite(local_mem_bw)) {
                Sys::sys_panic("system config: local-mem-bw x 1e9 is not "
                               "finite");
            }
        }
        if (j.contains("local-mem-latency")) {
            // R7-B fix: an unsigned member; a negative JSON integer used to wrap
            // to ~1.8e19 ns and stall every local-mem read forever.
            local_mem_latency =
                parse_non_negative_u64(j["local-mem-latency"],
                                       "local-mem-latency");  // ns
        }
        // Multi-user local-HBM bandwidth contention (LocalHbmBandwidthModel):
        // enabled by default; accepted as bool or int.  A non-positive
        // local-mem-bw would give the fluid model a zero full rate, which can
        // never drain a job -- auto-fall back to the legacy closed-form
        // behavior instead of stalling the simulation.
        this->hbm_bandwidth_contention = true;
        if (j.contains("hbm-bandwidth-contention")) {
            hbm_bandwidth_contention = parse_bool_or_int_flag(
                j["hbm-bandwidth-contention"], "hbm-bandwidth-contention");
        }
        if (this->local_mem_bw <= 0) {
            this->hbm_bandwidth_contention = false;
        }
        if (j.contains("roofline-enabled")) {
            if (parse_bool_or_int_flag(j["roofline-enabled"], "roofline-enabled")) {
                roofline_enabled = true;
                roofline = new Roofline(local_mem_bw, peak_perf);
            }
        }
        this->trace_enabled = false;
        if (j.contains("trace-enabled")) {
            this->trace_enabled =
                parse_bool_or_int_flag(j["trace-enabled"], "trace-enabled");
        }
        this->replay_only = false;
        if (j.contains("replay-only")) {
            this->replay_only =
                parse_bool_or_int_flag(j["replay-only"], "replay-only");
        }
        // Compute-cost backend keys (stage B; contract §3.2). The three keys
        // are parsed here; the conflict matrix and the die hardware load run
        // after the whole key-walk so they can see the fully parsed
        // roofline-enabled / replay-only / peak-perf / local-mem-bw state.
        this->compute_cost_model = ComputeCostModel::LegacyDefault;
        explicit_compute_cost_model_present = j.contains("compute-cost-model");
        if (explicit_compute_cost_model_present) {
            if (!j["compute-cost-model"].is_string()) {
                Sys::sys_panic(
                    "system config key 'compute-cost-model' must be a "
                    "string");
            }
            const string model = j["compute-cost-model"];
            if (model == "die") {
                explicit_compute_cost_model = ComputeCostModel::Die;
            } else if (model == "roofline") {
                explicit_compute_cost_model = ComputeCostModel::Roofline;
            } else if (model == "replay") {
                explicit_compute_cost_model = ComputeCostModel::Replay;
            } else {
                sys_panic("unknown value for compute-cost-model in sys input "
                          "file: '" + model + "' (legal: die | roofline | "
                          "replay)");
            }
        }
        die_model_profile_present = j.contains("die-model-profile");
        if (die_model_profile_present) {
            if (!j["die-model-profile"].is_string()) {
                Sys::sys_panic(
                    "system config key 'die-model-profile' must be a "
                    "string");
            }
            die_model_profile = j["die-model-profile"];
            if (die_model_profile != "mesh_die_v1" &&
                die_model_profile != "tx8_reference_v1") {
                sys_panic("unknown value for die-model-profile in sys input "
                          "file: '" + die_model_profile + "' (legal: "
                          "mesh_die_v1 | tx8_reference_v1)");
            }
        }
        die_model_config_explicit_present = j.contains("die-model-config");
        if (die_model_config_explicit_present) {
            if (!j["die-model-config"].is_string()) {
                Sys::sys_panic(
                    "system config key 'die-model-config' must be a string");
            }
            die_model_config_path = j["die-model-config"];
        }
        this->track_local_mem = false;
        if (j.contains("track-local-mem")) {
            this->track_local_mem =
                parse_bool_or_int_flag(j["track-local-mem"], "track-local-mem");
        }

        this->local_mem_trace_filename = "local_mem_trace";
        if (j.contains("local-mem-trace-filename")) {
            if (!j["local-mem-trace-filename"].is_string()) {
                Sys::sys_panic("system config key "
                               "'local-mem-trace-filename' must be a "
                               "string");
            }
            this->local_mem_trace_filename = j["local-mem-trace-filename"];
        }

        // The lookup setup parses the implementation lists (and the per-node
        // YAML); its arity/parse errors are plain exceptions that used to escape
        // through the same unguarded `new Sys` chain. Same fail-closed wrapper.
        collective_impl_lookup->setup_collective_impl_from_config(j);
    } catch (const std::exception& e) {
        sys_panic("invalid system config file " + name + ": " + e.what());
    }

    // Compute cost backend resolution (stage B; contract §3.2 conflict
    // matrix, fail-closed on every branch -- field parse order must never
    // select a backend silently).
    if (explicit_compute_cost_model_present &&
        explicit_compute_cost_model == ComputeCostModel::Die) {
        if (roofline_enabled) {
            sys_panic("system config conflict: compute-cost-model=die is "
                      "incompatible with roofline-enabled=1 (contract 3.2)");
        }
        if (replay_only) {
            sys_panic("system config conflict: compute-cost-model=die is "
                      "incompatible with replay-only=1 (contract 3.2)");
        }
        initialize_die_backend(name, die_model_config_explicit_present,
                               die_model_config_path, die_model_profile_present,
                               die_model_profile, peak_perf_key_present,
                               local_mem_bw_key_present,
                               hbm_contention_key_present);
    } else if (explicit_compute_cost_model_present &&
               explicit_compute_cost_model == ComputeCostModel::Roofline) {
        // Explicit legacy roofline backend: same dispatch state as the
        // historical roofline-enabled=1 configuration.
        compute_cost_model = ComputeCostModel::Roofline;
        if (!roofline_enabled) {
            roofline_enabled = true;
            roofline = new Roofline(local_mem_bw, peak_perf);
        }
    } else if (explicit_compute_cost_model_present) {
        // Explicit replay backend: the legacy no-roofline dispatch
        // (COMP nodes run through issue_replay semantics).
        compute_cost_model = ComputeCostModel::Replay;
        if (die_model_profile_present || die_model_config_explicit_present) {
            LoggerFactory::get_logger("system")->warn(
                "system config file {} carries die-model keys but "
                "compute-cost-model is not 'die'; they are ignored",
                name);
        }
    } else {
        // No compute-cost-model key: backward-compatible dispatch. The die
        // keys never select the die backend on their own.
        compute_cost_model = roofline_enabled ? ComputeCostModel::Roofline
                                              : ComputeCostModel::LegacyDefault;
        if (die_model_profile_present || die_model_config_explicit_present) {
            LoggerFactory::get_logger("system")->warn(
                "system config file {} carries die-model keys but no "
                "compute-cost-model=die; they are ignored",
                name);
        }
    }

    // Unknown-key audit (2026-10-01): warn (do NOT panic) on top-level keys
    // with no consumer. The whitelist below is the union of this function's
    // key-walk and the implementation-list parse in
    // CollectiveImplLookup::setup_collective_impl_from_config. A typo'd key
    // used to be dropped silently (fail-open on spelling). Warn only, because
    // examples/system/native_collectives/*.json legitimately carry a
    // consumer-less 'boost-mode' key and must keep loading. Rank 0 reports
    // once -- every rank parses the same file.
    static const char* known_system_keys[] = {
        "scheduling-policy", "inter-dimension-scheduling",
        "intra-dimension-scheduling", "collective-optimization",
        "local-reduction-delay", "active-chunks-per-dimension",
        "L", "o", "g", "G", "endpoint-delay", "model-shared-bus",
        "preferred-dataset-splits", "peak-perf", "local-mem-bw",
        "local-mem-latency", "hbm-bandwidth-contention", "roofline-enabled",
        "trace-enabled", "replay-only", "track-local-mem",
        "local-mem-trace-filename", "all-reduce-implementation",
        "all-reduce-implementation-custom", "reduce-scatter-implementation",
        "reduce-scatter-implementation-custom", "all-gather-implementation",
        "all-gather-implementation-custom", "all-to-all-implementation",
        "all-to-all-implementation-custom", "per-node-custom-implementation",
        "compute-cost-model", "die-model-profile", "die-model-config",
    };
    if (id == 0) {
        std::vector<std::string> unknown_keys;
        for (auto it = j.begin(); it != j.end(); ++it) {
            bool known = false;
            for (const char* known_key : known_system_keys) {
                if (it.key() == known_key) {
                    known = true;
                    break;
                }
            }
            if (!known) {
                unknown_keys.push_back(it.key());
            }
        }
        if (!unknown_keys.empty()) {
            std::string joined;
            for (size_t i = 0; i < unknown_keys.size(); i++) {
                if (i > 0) {
                    joined += ", ";
                }
                joined += "'" + unknown_keys[i] + "'";
            }
            LoggerFactory::get_logger("system")->warn(
                "system config file {} contains keys unknown to this "
                "simulation (they are ignored): {}",
                name, joined);
        }
    }

    inFile.close();
    return true;
}

void Sys::initialize_die_backend(
    const std::string& system_config_path,
    const bool explicit_config_present,
    const std::string& explicit_config_path,
    const bool profile_present,
    const std::string& profile,
    const bool peak_perf_key_present,
    const bool local_mem_bw_key_present,
    const bool hbm_contention_key_present) {
    compute_cost_model = ComputeCostModel::Die;

    // Resolve the die-model-config path (contract §3.2): the explicit key,
    // else resolved_die_model.json next to the system.json being parsed.
    string path = explicit_config_path;
    if (path.empty()) {
        const std::filesystem::path base(system_config_path);
        path = (base.parent_path() / "resolved_die_model.json").string();
    }
    die_model_config_path = path;

    // Fail-closed load: the file must exist and parse.
    std::ifstream in(path);
    if (!in.is_open()) {
        sys_panic("die backend requires die-model-config: cannot open " +
                  path + (explicit_config_present
                              ? " (from the die-model-config key)"
                              : " (default: resolved_die_model.json next to "
                                "the system config)"));
    }
    json die_cfg;
    try {
        in >> die_cfg;
    } catch (const std::exception& e) {
        sys_panic("failed to parse die-model-config file " + path + ": " +
                  e.what());
    }

    DieModel::Hardware hw;
    try {
        hw = DieModel::make_hardware(die_cfg);
    } catch (const std::exception& e) {
        LoggerFactory::get_logger("system")->critical(
            "die-model-config file {} rejected by the die cost library: {}",
            path, e.what());
        exit(1);
    }

    // Profile cross-check: the resolved file's own profile is what the cost
    // library runs with; an explicit system.json die-model-profile that
    // disagrees is a resolver/Config regression -- fail closed.
    if (profile_present &&
        profile != DieModel::to_string(hw.profile)) {
        sys_panic("system config die-model-profile '" + profile +
                  "' disagrees with the profile of " + path + " ('" +
                  DieModel::to_string(hw.profile) + "')");
    }

    // Derived-scalar consistency (contract §3.4): explicit peak-perf /
    // local-mem-bw must equal the library-derived PE_peak_die /
    // hbm_bandwidth_bytes -- the resolver guarantees this, so a mismatch is
    // a broken resolver or a hand-edited config and is refused (never
    // silently resolved to either side). Missing keys are filled FROM the
    // library so the legacy consumers (statistics denominators, the HBM
    // fluid model) keep working under the die backend.
    const auto same_scalar = [](const double a, const double b) {
        const double scale = std::max(1.0, std::max(std::abs(a), std::abs(b)));
        return std::abs(a - b) <= 1e-9 * scale;
    };
    const double derived_peak_flop_s = hw.pe_peak_die;
    const double derived_mem_bw_bytes_s = hw.hbm_bandwidth_bytes;
    if (peak_perf_key_present &&
        !same_scalar(peak_perf, derived_peak_flop_s)) {
        sys_panic("system config peak-perf (" + std::to_string(peak_perf) +
                  " FLOP/s) disagrees with the die-model-config derived "
                  "PE_peak_die (" +
                  std::to_string(derived_peak_flop_s) + " FLOP/s, from " +
                  path + "); the resolver must keep the two identical");
    }
    if (local_mem_bw_key_present &&
        !same_scalar(local_mem_bw, derived_mem_bw_bytes_s)) {
        sys_panic("system config local-mem-bw (" +
                  std::to_string(local_mem_bw) +
                  " byte/s) disagrees with the die-model-config derived "
                  "hbm_bandwidth_bytes (" +
                  std::to_string(derived_mem_bw_bytes_s) + " byte/s, from " +
                  path + "); the resolver must keep the two identical");
    }
    if (!peak_perf_key_present) {
        peak_perf = derived_peak_flop_s;
    }
    if (!local_mem_bw_key_present) {
        local_mem_bw = derived_mem_bw_bytes_s;
        // The key-walk's local_mem_bw <= 0 auto-disable of the HBM fluid
        // model ran before this fill; when the user spelled neither
        // local-mem-bw nor hbm-bandwidth-contention, restore the documented
        // default (contention on) so the die backend arbitrates memory as
        // designed instead of silently running the contention-off arm.
        if (!hbm_contention_key_present && local_mem_bw > 0) {
            hbm_bandwidth_contention = true;
        }
    }

    die_runtime = std::make_unique<DieModelRuntime>(std::move(hw));
}

Sys::DieModelRuntime::DieModelRuntime(DieModel::Hardware hww)
    : hardware(std::move(hww)), plan_cache(shared_die_plan_cache(hardware)) {}

Tick Sys::boostedTick() {
    Sys* ts = all_sys[0];
    if (ts == nullptr) {
        for (uint64_t i = 1; i < all_sys.size(); i++) {
            if (all_sys[i] != nullptr) {
                ts = all_sys[i];
                break;
            }
        }
    }
    timespec_t tmp = ts->comm_NI->sim_get_time();
    Tick tick = tmp.time_val / CLOCK_PERIOD;
    return tick;
}

[[noreturn]] void Sys::sys_panic(string msg) {
    auto logger = LoggerFactory::get_logger("system");
    logger->critical(msg);
    exit(1);
}

void Sys::call(EventType type, CallData* data) {
    // (2026-09-30 note) Deliberately empty: nothing registers Sys itself as
    // an event Callable today (verify with a grep for register_event(this)
    // in Sys.cc), so this override only satisfies the Callable interface.
    // If that ever changes, replace the body with sys_panic -- an empty
    // call would silently swallow the event.
    (void)type;
    (void)data;
}

void Sys::call_events() {
    const Tick now = Sys::boostedTick();
    auto event_list_it = event_queue.find(now);
    if (event_list_it == event_queue.end()) {
        // A previously cancelled last event may still have a backend alarm
        // queued. Its payload/list node is already gone, so this is a true
        // no-op rather than recreating an empty map bucket with operator[].
        return;
    }

    dispatching_events = true;
    dispatching_event_time = now;
    auto& scheduled_events = event_list_it->second.events;
    // The analytical frontend removes its registry entry before calling us.
    // Forget this weak outer identity before user callbacks can re-enter Sys.
    event_list_it->second.outer_alarm.reset();
    while (!scheduled_events.empty()) {
        // Pop before callback invocation. A re-entrant cancellation can thus
        // only remove a later pending node; it cannot free the CallData the
        // current callback is executing with.
        ScheduledEvent callable = std::move(scheduled_events.front());
        scheduled_events.pop_front();
        try {
            callable.callable->call(callable.event, callable.call_data);
        } catch (const std::exception& e) {
            auto logger = LoggerFactory::get_logger("system");
            logger->critical("warning! a callable callback threw an "
                             "exception; propagating after restoring the "
                             "dispatch state {}",
                             e.what());
            // Restore the dispatch invariants before propagating: a fail-loud
            // Workload exception must terminate the run instead of leaving the
            // node half-dispatched with the bucket still installed.
            dispatching_events = false;
            event_queue.erase(event_list_it);
            throw;
        }
    }
    dispatching_events = false;
    event_queue.erase(event_list_it);
}

void Sys::register_event(Callable* callable,
                         EventType event,
                         CallData* callData,
                         Tick delta_cycles) {
    try_register_event(callable, event, callData, delta_cycles);
}

SystemEventHandle Sys::register_event_cancellable(
    Callable* callable,
    EventType event,
    CallData* callData,
    Tick delta_cycles,
    EventDataCancellationCallback cancellation_callback) {
    if (delta_cycles == 0) {
        // A delta of 0 is ambiguous across invoke contexts: inside the
        // backend's invoke pass it means same-tick merge, outside it would
        // need the next tick -- two different calendars (orphaned events,
        // silently dropped wakeups) depending on the calling context.  The
        // only production caller (LocalHbmBandwidthModel) already clamps to
        // >= 1, so the guard costs nothing today and turns the ambiguous
        // request into a named error instead.
        sys_panic("register_event_cancellable requires delta >= 1 ns (a "
                  "delta of 0 is ambiguous across invoke contexts and was "
                  "never a supported request)");
    }
    if (next_cancellable_event_id == 0) {
        sys_panic("cancellable system event id space exhausted");
    }
    const uint64_t event_id = next_cancellable_event_id++;
    const Tick event_time = Sys::boostedTick() + delta_cycles;
    auto [event_list_it, should_schedule] = event_queue.try_emplace(event_time);
    event_list_it->second.events.push_back(
        {callable, event, callData, event_id, cancellation_callback});
    if (should_schedule) {
        timespec_t tmp;
        tmp.time_res = NS;
        tmp.time_val = delta_cycles;
        BasicEventHandlerData* data =
            new BasicEventHandlerData(id, EventType::CallEvents);
        data->sys_id = id;
        event_list_it->second.outer_alarm = comm_NI->sim_schedule_cancellable(
            tmp, &Sys::handleEvent, data, &cancel_call_events_alarm);
    }
    SystemEventHandle handle;
    handle.owner_ = this;
    handle.event_time_ = event_time;
    handle.event_id_ = event_id;
    return handle;
}

bool Sys::cancel_event(SystemEventHandle& handle) {
    if (!handle.valid() || handle.owner_ != this) {
        handle.reset();
        return false;
    }

    const auto event_list_it = event_queue.find(handle.event_time_);
    if (event_list_it == event_queue.end()) {
        handle.reset();
        return false;
    }
    for (auto event_it = event_list_it->second.events.begin();
         event_it != event_list_it->second.events.end(); ++event_it) {
        if (event_it->event_id != handle.event_id_) {
            continue;
        }
        CallData* const payload = event_it->call_data;
        const auto cancellation_callback = event_it->cancellation_callback;
        event_list_it->second.events.erase(event_it);
        handle.reset();
        if (event_list_it->second.events.empty() &&
            (!dispatching_events ||
             event_list_it->first != dispatching_event_time)) {
            static_cast<void>(
                comm_NI->sim_cancel_event(event_list_it->second.outer_alarm));
            event_queue.erase(event_list_it);
        }
        if (cancellation_callback != nullptr) {
            cancellation_callback(payload);
        }
        return true;
    }
    handle.reset();
    return false;
}

void Sys::try_register_event(Callable* callable,
                             EventType event,
                             CallData* callData,
                             Tick& delta_cycles) {
    // H3 (2026-09-30, mirrored from the joint repo): a delta==0 registration
    // outside the backend's invoke pass hits the EventQueue's same-tick
    // strict-increase guard and exits the whole run.  Online tick-end /
    // deferred commits issue collectives on this path, and a
    // communication_delay truncated to 0 by a fractional injection scale
    // (legal CLI combo, e.g. endpoint-delay 10 with --injection-scale 0.05)
    // makes the first collective packet fatal with a message unrelated to
    // its cause.  Same-tick merging stays byte-exact inside the invoke pass
    // (static dispatch), so only the forbidden context defers to the next
    // tick.
    if (delta_cycles == 0 && !comm_NI->backend_in_invoke_context()) {
        delta_cycles = 1;
    }
    auto event_time = Sys::boostedTick() + delta_cycles;
    auto [event_list_it, should_schedule] = event_queue.try_emplace(event_time);
    event_list_it->second.events.push_back(
        {callable, event, callData, 0, nullptr});
    if (should_schedule) {
        timespec_t tmp;
        tmp.time_res = NS;
        tmp.time_val = delta_cycles;
        BasicEventHandlerData* data =
            new BasicEventHandlerData(id, EventType::CallEvents);
        data->sys_id = id;
        comm_NI->sim_schedule(tmp, &Sys::handleEvent, data);
    }
    delta_cycles = 0;
    return;
}

void Sys::handleEvent(void* arg) {
    if (arg == nullptr) {
        return;
    }
    BasicEventHandlerData* ehd = (BasicEventHandlerData*)arg;
    int id = ehd->sys_id;
    EventType event = ehd->event;

    if (event == EventType::CallEvents) {
        // Analytical frontends cancel the matching outer alarm when the final
        // Sys event disappears. Legacy/fake frontends retain sim_schedule's
        // old behavior, so keep this fallback safe if such a stale alarm
        // arrives after its Sys (and HBM model) has gone away.
        if (id >= 0 && static_cast<size_t>(id) < all_sys.size() &&
            all_sys[id] != nullptr) {
            all_sys[id]->call_events();
        }
        delete ehd;
    } else if (event == EventType::RendezvousSend) {
        RendezvousSendData* rsd = (RendezvousSendData*)ehd;
        rsd->send.call(EventType::General, nullptr);
        delete rsd;
    } else if (event == EventType::RendezvousRecv) {
        RendezvousRecvData* rrd = (RendezvousRecvData*)ehd;
        rrd->recv.call(EventType::General, nullptr);
        delete rrd;
    } else if (event == EventType::PacketReceived) {
        RecvPacketEventHandlerData* rcehd = (RecvPacketEventHandlerData*)ehd;
        // Invariant (fail-loud guard, do NOT reorder the callbacks below):
        // rcehd->wlhd is constructed at exactly two sites -- Workload.cc
        // (sets workload, leaves custom_algorithm null) and CustomAlgorithm.cc
        // (sets custom_algorithm, leaves workload null) -- so at most one of
        // the two consumers may be set per message. BOTH consumers end by
        // deleting wlhd, so a message carrying both would be a use-after-free
        // on the second leg, and reordering the callbacks would only change
        // which object is dereferenced after the free. The five-argument
        // construction sites (Ring.cc/HalvingDoubling.cc/
        // DoubleBinaryTreeAllReduce.cc -- the trailing vnet/stream_id
        // arguments are signature-compatibility stubs) set owner only and
        // are unaffected.
        if (rcehd->workload && rcehd->custom_algorithm) {
            Sys::sys_panic(
                "PacketReceived message carries both workload and "
                "custom_algorithm consumers: the two construction sites are "
                "mutually exclusive by invariant, and both consumers delete "
                "the shared payload (guaranteed use-after-free). See "
                "Workload.cc and CustomAlgorithm.cc construction of "
                "RecvPacketEventHandlerData.");
        }
        if (rcehd->workload) {
            rcehd->workload->call(event, rcehd->wlhd);
        }
        if (rcehd->owner) {
            rcehd->owner->consume(rcehd);
        }
        if (rcehd->custom_algorithm) {
            rcehd->custom_algorithm->call(event, rcehd->wlhd);
        }
        delete rcehd;
    } else if (event == EventType::PacketSent) {
        SendPacketEventHandlerData* sehd = (SendPacketEventHandlerData*)ehd;
        sehd->callable->call(EventType::PacketSent, sehd->wlhd);
        delete sehd;
    }
}

LogicalTopology* Sys::get_logical_topology(ComType comm_type) {
    if (comm_type == ComType::All_Reduce) {
        return logical_topologies["AllReduce"];
    } else if (comm_type == ComType::All_to_All) {
        return logical_topologies["AllToAll"];
    } else if (comm_type == ComType::Reduce_Scatter) {
        return logical_topologies["ReduceScatter"];
    } else if (comm_type == ComType::All_Gather) {
        return logical_topologies["AllGather"];
    } else {
        sys_panic("no known logical topology for comm_type " +
                  std::to_string(static_cast<int>(comm_type)) +
                  " (expected All_Reduce/All_to_All/Reduce_Scatter/"
                  "All_Gather)");
        return nullptr;
    }
}

DataSet* Sys::generate_all_reduce(uint64_t size,
                                  vector<bool> involved_dimensions,
                                  CommunicatorGroup* communicator_group,
                                  int explicit_priority,
                                  uint64_t workload_node_id) {
    if (communicator_group == nullptr) {
        vector<CollectiveImpl*> implementation_per_dimension;
        implementation_per_dimension = collective_impl_lookup->get_collective_impl(ComType::All_Reduce, workload_node_id);
        return generate_collective(size, logical_topologies["AllReduce"],
                                   implementation_per_dimension,
                                   involved_dimensions, ComType::All_Reduce,
                                   explicit_priority, communicator_group);
    } else {
        CollectivePlan* plan =
            communicator_group->get_collective_plan(ComType::All_Reduce, workload_node_id);
        return generate_collective(
            size, plan->topology, plan->implementation_per_dimension,
            plan->dimensions_involved, ComType::All_Reduce, explicit_priority,
            communicator_group);
    }
}

DataSet* Sys::generate_all_to_all(uint64_t size,
                                  vector<bool> involved_dimensions,
                                  CommunicatorGroup* communicator_group,
                                  int explicit_priority,
                                  uint64_t workload_node_id) {
    if (communicator_group == nullptr) {
        vector<CollectiveImpl*> implementation_per_dimension;
        implementation_per_dimension = collective_impl_lookup->get_collective_impl(ComType::All_to_All, workload_node_id);
        return generate_collective(size, logical_topologies["AllToAll"],
                                   implementation_per_dimension,
                                   involved_dimensions, ComType::All_to_All,
                                   explicit_priority, communicator_group);
    } else {
        CollectivePlan* plan =
            communicator_group->get_collective_plan(ComType::All_to_All, workload_node_id);
        return generate_collective(
            size, plan->topology, plan->implementation_per_dimension,
            plan->dimensions_involved, ComType::All_to_All, explicit_priority,
            communicator_group);
    }
}

DataSet* Sys::generate_all_gather(uint64_t size,
                                  vector<bool> involved_dimensions,
                                  CommunicatorGroup* communicator_group,
                                  int explicit_priority,
                                  uint64_t workload_node_id) {
    if (communicator_group == nullptr) {
        vector<CollectiveImpl*> implementation_per_dimension;
        implementation_per_dimension = collective_impl_lookup->get_collective_impl(ComType::All_Gather, workload_node_id);
        return generate_collective(size, logical_topologies["AllGather"],
                                   implementation_per_dimension,
                                   involved_dimensions, ComType::All_Gather,
                                   explicit_priority, communicator_group);
    } else {
        CollectivePlan* plan =
            communicator_group->get_collective_plan(ComType::All_Gather, workload_node_id);
        return generate_collective(
            size, plan->topology, plan->implementation_per_dimension,
            plan->dimensions_involved, ComType::All_Gather, explicit_priority,
            communicator_group);
    }
}

DataSet* Sys::generate_reduce_scatter(uint64_t size,
                                      vector<bool> involved_dimensions,
                                      CommunicatorGroup* communicator_group,
                                      int explicit_priority,
                                      uint64_t workload_node_id) {
    if (communicator_group == nullptr) {
        vector<CollectiveImpl*> implementation_per_dimension;
        implementation_per_dimension = collective_impl_lookup->get_collective_impl(ComType::Reduce_Scatter, workload_node_id);
        return generate_collective(size, logical_topologies["ReduceScatter"],
                                   implementation_per_dimension,
                                   involved_dimensions, ComType::Reduce_Scatter,
                                   explicit_priority, communicator_group);
    } else {
        CollectivePlan* plan =
            communicator_group->get_collective_plan(ComType::Reduce_Scatter, workload_node_id);
        return generate_collective(
            size, plan->topology, plan->implementation_per_dimension,
            plan->dimensions_involved, ComType::Reduce_Scatter,
            explicit_priority, communicator_group);
    }
}

DataSet* Sys::generate_collective(
    uint64_t size,
    LogicalTopology* topology,
    vector<CollectiveImpl*> implementation_per_dimension,
    vector<bool> dimensions_involved,
    ComType collective_type,
    int explicit_priority,
    CommunicatorGroup* communicator_group) {
    // Status note (2026-09-26 review, replaces the bare upstream TODOs):
    // for custom collectives the chunk size is already determined and the
    // 'preferred-dataset-splits' JSON value would be redundant, but the
    // variable is deeply intertwined in this code path, so the JSON key
    // must stay and determine_chunk_size() still runs. Refactoring this
    // away is deferred to a coordinated upstream change.
    uint64_t chunk_size = determine_chunk_size(size, collective_type);
    uint64_t recommended_chunk_size = chunk_size;
    const double streams_exact = ((double)size) / chunk_size;
    // (2026-10-01) the int stream count has no domain guard: a uint64
    // payload over ~116 GB with the total-nodes chunk floor overflows int
    // (UB) and constructs DataSet with a negative count. Fail closed
    // instead; the reachable payloads are MB-scale collective buffers.
    if (!std::isfinite(streams_exact) ||
        streams_exact > static_cast<double>(std::numeric_limits<int>::max())) {
        Sys::sys_panic("collective size overflows the int stream count");
    }
    int streams = ceil(streams_exact);
    uint64_t remain_size;
    DataSet* dataset = new DataSet(streams);
    int pri = get_priority(explicit_priority);
    int count = 0;
    if (id == 0 &&
        inter_dimension_scheduling == InterDimensionScheduling::OfflineGreedy) {
        if (last_scheduled_collective != Sys::boostedTick()) {
            offline_greedy->reset_loads();
            last_scheduled_collective = Sys::boostedTick();
        }
    }

    if (implementation_per_dimension[0]->type == CollectiveImplType::CustomCollectiveImpl) {
        // For custom collective, we create a single stream covering the entire data size,
        // and ignore all the logic below.
        int pos_in_comm = id;
        if (communicator_group != nullptr) {
            pos_in_comm = communicator_group->get_position_in_group();
        }
        CollectivePhase phase = generate_collective_phase(
            collective_type,
            nullptr,
            size,
            // We use the variable queue_id to encode the position of this rank in the communication.
            pos_in_comm,
            // Below two are default values.
            RingTopology::Direction::Clockwise,
            implementation_per_dimension[0],
            communicator_group);
        list<CollectivePhase> vect;
        vect.push_back(phase);
        int stream_id = num_streams++;
        if (communicator_group != nullptr) {
            stream_id = communicator_group->num_streams++;
        }
        StreamBaseline* newStream =
            new StreamBaseline(this, dataset, stream_id, vect, pri);
        newStream->current_queue_id = -1;
        insert_into_ready_list(newStream);
        // Custom collectives only ever create this single stream, while the
        // DataSet was sized from the estimated stream count above. Align the
        // count or notify_stream_finished can never fire (hang + leaked
        // wrapper_map entry).
        dataset->total_streams = 1;
        return dataset;
    }

    // Every dimension loop below indexes dimensions_involved through
    // dim_mapper[dim], with dim_mapper sized from the topology. An
    // involved_dim attribute present but shorter than the physical dimension
    // count (the producer only pads four true entries when the attribute is
    // absent) would read the vector<bool> out of bounds; fail loud with the
    // actual length instead. The custom-collective path above never indexes
    // the vector and is therefore left unvalidated.
    if (dimensions_involved.size() <
        static_cast<size_t>(topology->get_num_of_dimensions())) {
        sys_panic("dimensions_involved has " +
                  std::to_string(dimensions_involved.size()) +
                  " entries but the collective topology has " +
                  std::to_string(topology->get_num_of_dimensions()) +
                  " dimensions");
    }
    // H4 (R7-A) defense in depth (2026-09-29): every dim loop below indexes
    // the queue levels (built from the PHYSICAL dimension count) with this
    // topology's dimension ids. The OfflineGreedy arm already rejects a
    // schedule/dimension mismatch; this guards the remaining arms against a
    // topology with more dimensions than physical queue levels (e.g. a hand
    // written shaped comm group), which read vLevels out of bounds. The
    // group constructor rejects the same shape at load time; this is the
    // fail-loud backstop at the point of use.
    if (static_cast<size_t>(topology->get_num_of_dimensions()) >
        vLevels->levels.size()) {
        sys_panic(
            "collective topology has " +
            std::to_string(topology->get_num_of_dimensions()) +
            " dimensions but the physical queue levels only cover " +
            std::to_string(vLevels->levels.size()) +
            " physical dimensions; issuing would index the queue levels out "
            "of bounds");
    }

    while (size > 0) {
        count++;

        vector<int> dim_mapper(topology->get_num_of_dimensions());
        iota(begin(dim_mapper), end(dim_mapper), 0);
        if (collective_type == ComType::All_Gather) {
            reverse(dim_mapper.begin(), dim_mapper.end());
        }

        if (inter_dimension_scheduling ==
            InterDimensionScheduling::RoundRobin) {
            // The RoundRobin cursor is shared by every collective of this
            // Sys but wraps against the dimension count of whichever
            // topology happens to run next. A collective on a
            // lower-dimensional topology (e.g. a 1-dim sub-cluster or shaped
            // communicator) entering with a cursor at or above that count
            // fails the wrap-around equality below, the cursor escapes
            // [0, dims) permanently, and the next rotate(begin + cursor)
            // steps past end() (iterator arithmetic UB). Fail loud while
            // the drift is still explainable instead.
            if (round_robin_inter_dimension_scheduler >=
                topology->get_num_of_dimensions()) {
                sys_panic(
                    "RoundRobin inter-dimension scheduler cursor (" +
                    std::to_string(round_robin_inter_dimension_scheduler) +
                    ") is out of range for a collective on a " +
                    std::to_string(topology->get_num_of_dimensions()) +
                    "-dimensional topology: mixing topologies of different "
                    "dimension counts lets the shared cursor escape its "
                    "wrap-around");
            }
            rotate(dim_mapper.begin(),
                   dim_mapper.begin() + round_robin_inter_dimension_scheduler,
                   dim_mapper.end());
            round_robin_inter_dimension_scheduler++;
            if (round_robin_inter_dimension_scheduler ==
                topology->get_num_of_dimensions()) {
                round_robin_inter_dimension_scheduler = 0;
            }
        } else if (collective_type != ComType::All_to_All &&
                   inter_dimension_scheduling ==
                       InterDimensionScheduling::OfflineGreedy) {
            uint64_t prev_size = size;
            dim_mapper = offline_greedy->get_chunk_scheduling(
                communicator_group == nullptr ? 0
                                              : communicator_group->get_id(),
                communicator_group == nullptr
                    ? num_streams
                    : communicator_group->num_streams,
                communicator_group == nullptr
                    ? all_sys.size()
                    : communicator_group->involved_NPUs.size(),
                size, recommended_chunk_size, dimensions_involved,
                inter_dimension_scheduling, collective_type);
            // F1 fix (2026-09-28): get_chunk_scheduling always returns one
            // entry per physical dimension (sys->physical_dims.size()), each
            // valued in [0, physical_dims). The phase loops below walk
            // topology->get_num_of_dimensions() entries and index both
            // dimension tables (the topology's dimensions and
            // implementation_per_dimension) through those values, so the
            // schedule only lines up with this collective when the counts
            // match:
            // - a whole-cluster collective whose implementation list built
            //   fewer topology dimensions than physical dimensions (C < P;
            //   GeneralComplexTopology only rejects C > P, and the
            //   group == nullptr path carries the 4x-true default
            //   dimensions_involved, so the length guard inside
            //   get_chunk_scheduling cannot see the mismatch) would index
            //   the topology/implementation tables out of bounds, and
            // - a shaped communicator with more dimensions than physical
            //   dimensions (S > P) would read the schedule vector itself
            //   out of bounds.
            // Fail loud with both counts here; the group != nullptr
            // whole-cluster mismatch already fails loud inside
            // get_chunk_scheduling via the same dimensions_involved length
            // guard, so both paths now share one failure surface.
            if (dim_mapper.size() !=
                static_cast<size_t>(topology->get_num_of_dimensions())) {
                sys_panic(
                    "OfflineGreedy returned an inter-dimension schedule of " +
                    to_string(dim_mapper.size()) +
                    " physical dimensions (entries valued in [0, " +
                    to_string(dim_mapper.size()) +
                    ")) for a collective whose dimension tables hold " +
                    to_string(topology->get_num_of_dimensions()) +
                    " dimensions: consuming it would index the topology or "
                    "implementation tables out of bounds");
            }
            chunk_size = prev_size - size;
        }

        if (collective_type == ComType::All_to_All ||
            inter_dimension_scheduling !=
                InterDimensionScheduling::OfflineGreedy) {
            if (chunk_size > size) {
                size = 0;
            } else {
                size -= chunk_size;
            }
        }
        remain_size = chunk_size;
        list<CollectivePhase> vect;

        if (collective_type != ComType::All_Reduce ||
            collectiveOptimization == CollectiveOptimization::Baseline) {
            for (int dim = 0; dim < topology->get_num_of_dimensions(); dim++) {
                if (topology->get_num_of_nodes_in_dimension(dim_mapper[dim]) ==
                        1 ||
                    !dimensions_involved[dim_mapper[dim]]) {
                    continue;
                }
                pair<int, RingTopology::Direction> queue =
                    vLevels->get_next_queue_at_level(dim_mapper[dim]);
                CollectivePhase phase = generate_collective_phase(
                    collective_type,
                    topology->get_basic_topology_at_dimension(dim_mapper[dim],
                                                              collective_type),
                    remain_size, queue.first, queue.second,
                    implementation_per_dimension[dim_mapper[dim]]);
                vect.push_back(phase);
                remain_size = phase.final_data_size;
            }
        } else if (inter_dimension_scheduling ==
                   InterDimensionScheduling::OfflineGreedy) {
            int dim = 0;

            // Create collective phase for each dimension in ascending order.
            for (dim = 0; dim < topology->get_num_of_dimensions(); dim++) {
                if (topology->get_num_of_nodes_in_dimension(dim_mapper[dim]) ==
                        1 ||
                    !dimensions_involved[dim_mapper[dim]]) {
                    continue;
                }
                pair<int, RingTopology::Direction> queue =
                    vLevels->get_next_queue_at_level_first(dim_mapper[dim]);
                CollectivePhase phase = generate_collective_phase(
                    ComType::Reduce_Scatter,
                    topology->get_basic_topology_at_dimension(
                        dim_mapper[dim], ComType::Reduce_Scatter),
                    remain_size, queue.first, queue.second,
                    implementation_per_dimension[dim_mapper[dim]]);
                vect.push_back(phase);
                remain_size = phase.final_data_size;
            }
            dim--;

            // Create collective phases for each dimension in descending order.
            for (; dim >= 0; dim--) {
                if (topology->get_num_of_nodes_in_dimension(dim_mapper[dim]) ==
                        1 ||
                    !dimensions_involved[dim_mapper[dim]]) {
                    continue;
                }
                pair<int, RingTopology::Direction> queue =
                    vLevels->get_next_queue_at_level_last(dim_mapper[dim]);
                CollectivePhase phase = generate_collective_phase(
                    ComType::All_Gather,
                    topology->get_basic_topology_at_dimension(
                        dim_mapper[dim], ComType::All_Gather),
                    remain_size, queue.first, queue.second,
                    implementation_per_dimension[dim_mapper[dim]]);
                vect.push_back(phase);
                remain_size = phase.final_data_size;
            }
        } else {
            // In this branch, and the branch directly above, a collective
            // visits each dimension (excluding the last dimension) twice.
            // Specifically, for example, in 2D AllReduce, there would be 3
            // collective phases: Phase 0: Reduce Scatter in dim 0, Phase 1: All
            // Reduce in dim 1, Phase 2: All Gather in dim 0 Similarly, in 3D
            // AllReduce, there would be 5 collective phases: RS in dim 0, RS in
            // dim 1, AR in dim 2, AG in dim 1, AG in dim 0. Currently, queues
            // are allocated per dimension. If we allocate all queues in a
            // dimension to both phases of a single dimension, a race / deadlock
            // condition may occur. Therefore, in these cases, we have to
            // allocate half of the queues to the first phase, and the remaining
            // half to the second phase. (For example, in the above 2D case, if
            // we have 4 queues per dim, queues 0~1 are allocated to phase 0,
            // queues 2~3 are allocated to phase 2. For details, refer to
            // https://github.com/astra-sim/astra-sim/issues/137 and the linked
            // document.

            int dim = 0;
            int last_active_dim = 0;
            for (dim = 0; dim < topology->get_num_of_dimensions(); dim++) {
                if (topology->get_num_of_nodes_in_dimension(dim_mapper[dim]) !=
                        1 &&
                    dimensions_involved[dim_mapper[dim]]) {
                    last_active_dim = dim;
                }
            }

            // Create collective phase for each dimension, excluding the last
            // dimension, in ascending order.
            for (dim = 0; dim < last_active_dim; dim++) {
                if (topology->get_num_of_nodes_in_dimension(dim_mapper[dim]) ==
                        1 ||
                    !dimensions_involved[dim_mapper[dim]]) {
                    continue;
                }
                // Allocate the first half of queues available to this
                // dimension.
                pair<int, RingTopology::Direction> queue =
                    vLevels->get_next_queue_at_level_first(dim_mapper[dim]);
                CollectivePhase phase = generate_collective_phase(
                    ComType::Reduce_Scatter,
                    topology->get_basic_topology_at_dimension(
                        dim_mapper[dim], ComType::Reduce_Scatter),
                    remain_size, queue.first, queue.second,
                    implementation_per_dimension[dim_mapper[dim]]);
                vect.push_back(phase);
                remain_size = phase.final_data_size;
            }
            // (2026-09-30) The back-scan while-loop that used to live
            // here was unreachable: the for-loop above exits with
            // dim == last_active_dim, and last_active_dim by construction
            // satisfies (involved && nodes > 1), so the loop condition was
            // false on its very first evaluation in every reachable state.


            // The last dimension is the 'turning point'. Only one collective
            // phase is created.
            if (dimensions_involved[dim_mapper[dim]] &&
                topology->get_num_of_nodes_in_dimension(dim_mapper[dim]) > 1) {
                // Despite only one collective phase being allocated to the last
                // dimension, we only allocate half of the queues available to
                // this dimension. This is because we want to match the number
                // of queues allocated to each collective phase. Processing
                // phases for this dim in n parallel queues, and queueing the
                // next phases in n/2 parallel queues could cause another
                // deadlock. Refer to the PR #135 for more details.
                pair<int, RingTopology::Direction> queue =
                    vLevels->get_next_queue_at_level_first(dim_mapper[dim]);
                CollectivePhase phase = generate_collective_phase(
                    ComType::All_Reduce,
                    topology->get_basic_topology_at_dimension(
                        dim_mapper[dim], ComType::All_Reduce),
                    remain_size, queue.first, queue.second,
                    implementation_per_dimension[dim_mapper[dim]]);
                vect.push_back(phase);
                remain_size = phase.final_data_size;
            }
            dim--;

            // Create collective phases for each dimension, excluding the last
            // dimension, in descending order.
            for (; dim >= 0; dim--) {
                if (topology->get_num_of_nodes_in_dimension(dim_mapper[dim]) ==
                        1 ||
                    !dimensions_involved[dim_mapper[dim]]) {
                    continue;
                }
                // Allocate the second half of queues available to this
                // dimension.
                pair<int, RingTopology::Direction> queue =
                    vLevels->get_next_queue_at_level_last(dim_mapper[dim]);
                CollectivePhase phase = generate_collective_phase(
                    ComType::All_Gather,
                    topology->get_basic_topology_at_dimension(
                        dim_mapper[dim], ComType::All_Gather),
                    remain_size, queue.first, queue.second,
                    implementation_per_dimension[dim_mapper[dim]]);
                vect.push_back(phase);
                remain_size = phase.final_data_size;
            }
        }
        if (vect.size() > 0) {
            // Cross-rank precondition (documented 2026-10-01, upstream
            // inherited): num_streams is a PER-RANK counter, and the
            // backend matches packets by (tag built from this id, src, dst,
            // count, chunk). Two same-type same-size collectives of one
            // group interleave consistently ONLY if every rank issues its
            // collectives in the same global order. The online batch path
            // guarantees this by construction (layer deps + one global
            // commit order); a hand-written asymmetric static ET could
            // diverge per-rank stream ids and stall or cross-deliver.
            // Do not loosen the emission-order guarantee without changing
            // this to a globally unique tag source.
            int stream_id = num_streams++;
            if (communicator_group != nullptr) {
                stream_id = communicator_group->num_streams++;
            }
            StreamBaseline* newStream =
                new StreamBaseline(this, dataset, stream_id, vect, pri);
            newStream->current_queue_id = -1;
            insert_into_ready_list(newStream);
        } else {
            // Degenerate configuration: every involved dimension collapsed
            // to a single node, so no stream can be created even though
            // size > 0 sized the DataSet to >= 1 stream. Without aligning
            // the count, notify_stream_finished can never fire and the
            // workload waits for CollectiveCommunicationFinished forever
            // (the same hang mode the custom-collective path's
            // total_streams alignment above guards against). total_streams
            // = 0 routes this into DataSet::set_notifier's zero-stream
            // immediate completion.
            dataset->active = false;
            dataset->total_streams = 0;
            break;
        }
    }
    if (dataset->active) {
        dataset->total_streams = count;
    }
    return dataset;
}

CollectivePhase Sys::generate_collective_phase(
    ComType collective_type,
    BasicLogicalTopology* topology,
    uint64_t data_size,
    int queue_id,
    RingTopology::Direction direction,
    CollectiveImpl* collective_impl,
    CommunicatorGroup* comm_group) {
    // The All-Reduce split-phase paths (collective-optimization: localBWAware
    // and inter-dimension-scheduling: OfflineGreedy) reuse the All-Reduce
    // implementation for their Reduce_Scatter/All_Gather phases, so the
    // whitelist check inside CollectiveImplLookup::get_collective_impl --
    // which validated only the original collective type -- never sees the
    // rewritten phase type. Re-validate the pairing against the phase type
    // here, at the single choke point every collective phase is built
    // through; every configuration the lookup accepts also passes here.
    CollectiveImplLookup::ensure_native_impl_supports(collective_type,
                                                      collective_impl->type);
    if (collective_impl->type == CollectiveImplType::Ring ||
        collective_impl->type == CollectiveImplType::OneRing) {
        CollectivePhase vn(this, queue_id,
                           new Ring(collective_type, id,
                                    (RingTopology*)topology, data_size,
                                    direction));
        return vn;
    } else if (collective_impl->type == CollectiveImplType::Direct ||
               collective_impl->type == CollectiveImplType::OneDirect) {
        CollectivePhase vn(this, queue_id,
                           new AllToAll(collective_type,
                                        ((DirectCollectiveImpl*)collective_impl)
                                            ->direct_collective_window,
                                        id, (RingTopology*)topology, data_size,
                                        direction));
        return vn;
    } else if (collective_impl->type == CollectiveImplType::DoubleBinaryTree) {
        auto* btree = dynamic_cast<BinaryTree*>(topology);
        if (!btree) {
            Sys::sys_panic("DoubleBinaryTree requires a BinaryTree logical "
                           "topology");
        }
        CollectivePhase vn(this, queue_id,
                           new DoubleBinaryTreeAllReduce(id, btree, data_size));
        return vn;
    } else if (collective_impl->type == CollectiveImplType::HalvingDoubling ||
               collective_impl->type ==
                   CollectiveImplType::OneHalvingDoubling) {
        CollectivePhase vn(this, queue_id,
                           new HalvingDoubling(collective_type, id,
                                               (RingTopology*)topology,
                                               data_size));
        return vn;
    } else if (collective_impl->type == CollectiveImplType::CustomCollectiveImpl) {
        string filename = ((CustomCollectiveImpl*)collective_impl)->filename;
        CollectivePhase vn(this, 0, new CustomAlgorithm(filename, id, queue_id, comm_group));
        return vn;
    } else {
        LoggerFactory::get_logger("system")->critical(
            "Error: No known collective implementation for collective phase");
        exit(1);
    }
}

uint64_t Sys::determine_chunk_size(uint64_t& size, ComType type) {
    if (size == 0) {
        // A zero-size collective has no chunk to split: hand back a unit
        // chunk so the caller's ceil division yields zero streams instead of
        // a NaN/inf from dividing by the chunk size.
        return 1;
    }
    if (preferred_dataset_splits <= 0) {
        // R7-W1: this branch is only reachable for a value that slipped
        // past the parse-time validation (the config reader rejects
        // non-positive values since the M6 fix); a missing key defaults to
        // 1 and never lands here. Keep the fallback one-chunk behavior as
        // the belt-and-braces guard: one chunk covering the whole size.
        return size;
    }
    uint64_t chunk_size = size / preferred_dataset_splits;
    if (chunk_size == 0 && type == ComType::All_Gather) {
        // 0 < size < splits: integer division floors to 0, which would make
        // the caller compute ceil(size/0) = +inf and loop forever issuing
        // streams. Fall back to a single chunk covering the whole size.
        // All_Gather only: its ring message IS the chunk (no per-node
        // division), so a below-nodes chunk stays nonzero. Every other type
        // divides the chunk by nodes_in_ring downstream and must fall
        // through to the total_nodes floor below, or a 1-byte sync
        // collective divides down to a zero-byte send (the sibling repos
        // guard All_Gather only, for the same reason).
        return size;
    }
    // We want the collective size to have minimum size, otherwise, there is a
    // possibility of size overflow due to further dividing it to more
    // fine-grained messages
    if (type != ComType::All_Gather && this->total_nodes > chunk_size) {
        chunk_size = this->total_nodes;
        size = preferred_dataset_splits * chunk_size;
    }
    // Upstream-inherited sizing semantics, documented 2026-10-01 (frozen
    // timing baseline -- do NOT "fix" without a cross-repo decision):
    // (a) when the total_nodes floor fires, `size` is rounded UP to
    //     splits x chunk (small control collectives can grow up to
    //     splits x total_nodes bytes);
    // (b) the LAST stream of a non-divisible collective carries a FULL
    //     chunk_size, not the remainder (generate_collective clamps its
    //     remaining_data_size only after stream_count = ceil(size/chunk)
    //     was decided), so injected NoC bytes overshoot by up to one
    //     chunk (~+1/6 at splits=6 on non-divisible sizes). Every rank
    //     overshoots identically, so pairing/completion are unaffected.
    return chunk_size;
}

int Sys::get_priority(int explicit_priority) {
    if (scheduling_policy == SchedulingPolicy::LIFO) {
        return priority_counter++;
    } else if (scheduling_policy == SchedulingPolicy::FIFO) {
        return priority_counter--;
    } else if (scheduling_policy == SchedulingPolicy::EXPLICIT) {
        return explicit_priority;
    }

    // should not reach here (2026-09-30): the old bare assert + exit(-1)
    // was SILENT in Release builds. Name the unreachable scheduler policy.
    sys_panic("Sys::get_priority reached an unknown scheduling policy");
}

void Sys::insert_into_ready_list(BaseStream* stream) {
    insert_stream(&ready_list, stream);
    scheduler_unit->notify_stream_added_into_ready_list();
}

void Sys::insert_stream(list<BaseStream*>* queue, BaseStream* baseStream) {
    list<BaseStream*>::iterator it = queue->begin();
    if (intra_dimension_scheduling == IntraDimensionScheduling::FIFO ||
        baseStream->current_queue_id < 0 ||
        baseStream->current_com_type == ComType::All_to_All ||
        baseStream->current_com_type == ComType::All_Reduce) {
        while (it != queue->end()) {
            if ((*it)->initialized == true) {
                advance(it, 1);
                continue;
            } else if ((*it)->priority >= baseStream->priority) {
                advance(it, 1);
                continue;
            } else {
                break;
            }
        }
    } else if (intra_dimension_scheduling == IntraDimensionScheduling::RG) {
        ComType one_to_last = ComType::None;
        ComType last = ComType::None;
        while (it != queue->end()) {
            one_to_last = last;
            last = (*it)->current_com_type;
            if ((*it)->initialized == true) {
                advance(it, 1);
                if (it != queue->end() && (*it)->initialized == false) {
                    one_to_last = last;
                    last = (*it)->current_com_type;
                    advance(it, 1);
                }
                continue;
            } else if ((*it)->priority > baseStream->priority) {
                advance(it, 1);
                continue;
            } else if ((last == ComType::Reduce_Scatter &&
                        one_to_last == ComType::All_Gather) ||
                       (last == ComType::All_Gather &&
                        one_to_last == ComType::Reduce_Scatter)) {
                advance(it, 1);
                continue;
            } else {
                break;
            }
        }
    } else if (intra_dimension_scheduling ==
               IntraDimensionScheduling::SmallestFirst) {
        if (baseStream->phases_to_go.size() == 1) {
            it = queue->end();
        }
        while (it != queue->end()) {
            if ((*it)->initialized == true) {
                advance(it, 1);
                continue;
            } else if (max((*it)->my_current_phase.initial_data_size,
                           (*it)->my_current_phase.final_data_size) <
                       max(baseStream->my_current_phase.initial_data_size,
                           baseStream->my_current_phase.final_data_size)) {
                advance(it, 1);
                continue;
            } else {
                break;
            }
        }
    } else if (intra_dimension_scheduling ==
               IntraDimensionScheduling::LessRemainingPhaseFirst) {
        while (it != queue->end()) {
            if ((*it)->initialized == true) {
                advance(it, 1);
                continue;
            } else if ((*it)->phases_to_go.size() <
                       baseStream->phases_to_go.size()) {
                advance(it, 1);
                continue;
            } else {
                break;
            }
        }
    }
    queue->insert(it, baseStream);
}

void Sys::ask_for_schedule(int max) {
    if (ready_list.size() == 0) {
        return;
    }
    int top = ready_list.front()->stream_id;
    uint64_t min = ready_list.size();
    if (min > max) {
        min = static_cast<uint64_t>(max);
    }
    for (auto& sys : all_sys) {
        // R1-D-L4: a null slot (torn-down rank) makes the all-rank front
        // scan unevaluable; the all-or-nothing contract returns instead of
        // dereferencing it.
        if (sys == nullptr || sys->ready_list.size() == 0 ||
            sys->ready_list.front()->stream_id != top) {
            return;
        }
        if (sys->ready_list.size() < min) {
            min = sys->ready_list.size();
        }
    }
    for (auto& sys : all_sys) {
        sys->schedule(min);
    }
    return;
}

void Sys::schedule(int num) {
    int ready_list_size = ready_list.size();
    int counter = min(num, ready_list_size);
    while (counter > 0) {
        // NOTE: proceed_to_next_vnet_baseline deletes the stream when its
        // last phase just completed, so ready_list.front() must not be
        // dereferenced after it returns. The post-proceed "should not
        // happen!" panic this loop used to carry checked
        // current_queue_id == -1 on exactly that possibly-dangling front,
        // but the state it guarded is unreachable: proceed assigns
        // current_queue_id from the next phase's queue id only after its
        // own fail-loud guard rejects a negative queue id (a phase handed a
        // queue that does not exist, e.g. a zero-queue dimension), so a
        // surviving stream always carries queue id >= 0, and a stream with
        // no phases never reaches this loop because generate_collective
        // enqueues only vect.size() > 0 streams. The queue-id invariant is
        // therefore enforced where the value is produced (check before use
        // inside proceed), not re-read here through a dangling pointer.
        // INVARIANT (2026-09-30 note): pop-after-proceed is safe only while
        // a stream's first init() never synchronously completes that same
        // stream -- every current algorithm reaches the event loop through
        // an asynchronous register_event before exit() can run. If an
        // algorithm ever completes inside init(), the nested
        // notify_stream_removed -> schedule would double-proceed the stream
        // still sitting at the front of ready_list.
        proceed_to_next_vnet_baseline((StreamBaseline*)ready_list.front());

        ready_list.pop_front();
        counter--;
        first_phase_streams++;
        total_running_streams++;
    }
}

void Sys::proceed_to_next_vnet_baseline(StreamBaseline* stream) {
    int previous_vnet = stream->current_queue_id;
    if (stream->steps_finished == 1) {
        first_phase_streams--;
    }
    if (stream->my_current_phase.algorithm != nullptr) {
        // NOTE: this delete can run on the algorithm's own call stack (Ring /
        // HalvingDoubling reach it from exit() via iteratable()); it is safe
        // today only because every caller in that chain returns right after
        // without touching members and this function overwrites the pointer
        // below before any later use. Treat an algorithm as invalid the
        // moment its exit() runs.
        delete stream->my_current_phase.algorithm;
    }
    if (stream->phases_to_go.size() == 0) {
        stream->dataset->notify_stream_finished((StreamStat*)stream);
    }
    if (stream->current_queue_id >= 0 && stream->my_current_phase.enabled) {
        list<BaseStream*>& target =
            active_Streams.at(stream->my_current_phase.queue_id);
        // Erase by pointer identity, not by stream_id: group-stream ids and
        // the global Sys counter are independent bands that can coincide
        // (CommunicatorGroup::set_id), so an id match can hit another live
        // stream, leave this stream's own node in the list, and turn it
        // into a dangling entry once this stream is deleted below.
        for (list<BaseStream*>::iterator it = target.begin();
             it != target.end(); ++it) {
            if (*it == stream) {
                target.erase(it);
                break;
            }
        }
    }
    if (stream->phases_to_go.size() == 0) {
        total_running_streams--;
        if (previous_vnet >= 0) {
            scheduler_unit->notify_stream_removed(previous_vnet);
        }
        delete stream;
        return;
    }
    stream->steps_finished++;
    if (stream->phases_to_go.front().queue_id < 0) {
        // Fail loud instead of the old silent `queue_id *= -1` hot fix: a
        // negative queue id means this phase was handed a queue that does
        // not exist (QueueLevelHandler::get_next_queue_id returns -1 for a
        // dimension with zero queues, e.g. a queues-per-dim entry of 0).
        // Flipping the sign would silently occupy an unrelated queue of
        // another dimension -- the cross-dimension queue-sharing race
        // documented in the issue #137 comment in generate_collective.
        Sys::sys_panic(
            "negative queue id (" +
            std::to_string(stream->phases_to_go.front().queue_id) +
            ") in a collective phase: the phase was assigned a queue that "
            "does not exist. This usually comes from a dimension configured "
            "with zero queues (num-queues-per-dim / queues-per-dim entry of "
            "0); fix the system config so every involved dimension has at "
            "least one queue.");
    }
    stream->current_queue_id = stream->phases_to_go.front().queue_id;
    stream->current_com_type = stream->phases_to_go.front().comm_type;

    CollectivePhase vi = stream->phases_to_go.front();
    stream->my_current_phase = vi;
    stream->phases_to_go.pop_front();
    stream->initialized = false;

    if (stream->my_current_phase.enabled) {
        insert_stream(&active_Streams[stream->current_queue_id], stream);
    }

    stream->state = StreamState::Ready;

    if (previous_vnet >= 0) {
        scheduler_unit->notify_stream_removed(previous_vnet);
    }
    scheduler_unit->notify_stream_added(stream->current_queue_id);
}

int Sys::front_end_sim_send(Tick delay,
                            void* buffer,
                            uint64_t count,
                            int type,
                            int dst,
                            int tag,
                            sim_request* request,
                            Sys::FrontEndSendRecvType send_type,
                            void (*msg_handler)(void* fun_arg),
                            void* fun_arg) {
    if (send_type == Sys::FrontEndSendRecvType::NATIVE) {
        tag = tag % (Sys::FrontEndSendRecvType::COLLECTIVE -
                     Sys::FrontEndSendRecvType::NATIVE) +
              Sys::FrontEndSendRecvType::NATIVE;
    } else if (send_type == Sys::FrontEndSendRecvType::COLLECTIVE) {
        tag = tag % (Sys::FrontEndSendRecvType::RENDEZVOUS -
                     Sys::FrontEndSendRecvType::COLLECTIVE) +
              Sys::FrontEndSendRecvType::COLLECTIVE;
    } else {
        sys_panic("A type of RENDZVOUS should never issued in frontend");
    }
    if (rendezvous_enabled) {
        return rendezvous_sim_send(delay, buffer, count, type, dst, tag,
                                   request, msg_handler, fun_arg);
    } else {
        return sim_send(delay, buffer, count, type, dst, tag, request,
                        msg_handler, fun_arg);
    }
}

int Sys::front_end_sim_recv(Tick delay,
                            void* buffer,
                            uint64_t count,
                            int type,
                            int src,
                            int tag,
                            sim_request* request,
                            Sys::FrontEndSendRecvType recv_type,
                            void (*msg_handler)(void* fun_arg),
                            void* fun_arg) {
    if (recv_type == Sys::FrontEndSendRecvType::NATIVE) {
        tag = tag % (Sys::FrontEndSendRecvType::COLLECTIVE -
                     Sys::FrontEndSendRecvType::NATIVE) +
              Sys::FrontEndSendRecvType::NATIVE;
    } else if (recv_type == Sys::FrontEndSendRecvType::COLLECTIVE) {
        tag = tag % (Sys::FrontEndSendRecvType::RENDEZVOUS -
                     Sys::FrontEndSendRecvType::COLLECTIVE) +
              Sys::FrontEndSendRecvType::COLLECTIVE;
    } else {
        sys_panic("A type of RENDZVOUS should never issued in frontend");
    }
    if (rendezvous_enabled) {
        return rendezvous_sim_recv(delay, buffer, count, type, src, tag,
                                   request, msg_handler, fun_arg);
    } else {
        return sim_recv(delay, buffer, count, type, src, tag, request,
                        msg_handler, fun_arg);
    }
}

int Sys::rendezvous_sim_send(Tick delay,
                             void* buffer,
                             uint64_t count,
                             int type,
                             int dst,
                             int tag,
                             sim_request* request,
                             void (*msg_handler)(void* fun_arg),
                             void* fun_arg) {
    if (tag >= Sys::FrontEndSendRecvType::RENDEZVOUS) {
        sys_panic("tag is bigger than RENDEZVOUS_COMM_TAG_OFFSET, \
        which means it might be mistakenly used as a rendezvous tag.");
    }
    RendezvousSendData* rsd =
        new RendezvousSendData(id, this, buffer, count, type, dst, tag,
                               *request, msg_handler, fun_arg);
    sim_request newReq = *request;
    // M2 fix: see g_rendezvous_handshake_buffer above.
    uint64_t rendevouz_size = 8192;
    newReq.dstRank = request->srcRank;
    newReq.srcRank = request->dstRank;
    int newTag = tag + Sys::FrontEndSendRecvType::RENDEZVOUS;
    newReq.tag = newTag;
    sim_recv(delay, g_rendezvous_handshake_buffer, rendevouz_size, type, dst,
             newTag, &newReq, &Sys::handleEvent, rsd);
    return 1;
}

int Sys::rendezvous_sim_recv(Tick delay,
                             void* buffer,
                             uint64_t count,
                             int type,
                             int src,
                             int tag,
                             sim_request* request,
                             void (*msg_handler)(void* fun_arg),
                             void* fun_arg) {
    if (tag >= Sys::FrontEndSendRecvType::RENDEZVOUS) {
        sys_panic("tag is bigger than RENDEZVOUS_COMM_TAG_OFFSET, \
        which means it might be mistakenly used as a rendezvous tag.");
    }
    RendezvousRecvData* rrd =
        new RendezvousRecvData(id, this, buffer, count, type, src, tag,
                               *request, msg_handler, fun_arg);
    sim_request newReq = *request;
    uint64_t rendevouz_size = 8192;
    newReq.dstRank = request->srcRank;
    newReq.srcRank = request->dstRank;
    int newTag = tag + Sys::FrontEndSendRecvType::RENDEZVOUS;
    newReq.tag = newTag;
    sim_send(delay, g_rendezvous_handshake_buffer, rendevouz_size, type, src,
             newTag, &newReq, &Sys::handleEvent, rrd);
    return 1;
}

int Sys::sim_send(Tick /*delay -- kept for the unchanged call sites; every
                     caller reaches here with zero delay*/,
                  void* buffer,
                  uint64_t count,
                  int type,
                  int dst,
                  int tag,
                  sim_request* request,
                  void (*msg_handler)(void* fun_arg),
                  void* fun_arg) {
    comm_NI->sim_send(buffer, count, type, dst, tag, request, msg_handler,
                      fun_arg);
    return 1;
}

int Sys::sim_recv(Tick /*delay -- kept for the unchanged call sites; every
                     caller reaches here with zero delay*/,
                  void* buffer,
                  uint64_t count,
                  int type,
                  int src,
                  int tag,
                  sim_request* request,
                  void (*msg_handler)(void* fun_arg),
                  void* fun_arg) {
    comm_NI->sim_recv(buffer, count, type, src, tag, request, msg_handler,
                      fun_arg);
    return 1;
}
}  // namespace AstraSim
