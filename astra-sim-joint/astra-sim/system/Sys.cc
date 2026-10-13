/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/Sys.hh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <utility>

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/BaseStream.hh"
#include "astra-sim/system/CollectivePlan.hh"
#include "astra-sim/system/DataSet.hh"
#include "astra-sim/system/MemBus.hh"
#include "astra-sim/system/MemEventHandlerData.hh"
#include "astra-sim/system/QueueLevels.hh"
#include "astra-sim/system/RendezvousRecvData.hh"
#include "astra-sim/system/RendezvousSendData.hh"
#include "astra-sim/system/SendPacketEventHandlerData.hh"
#include "astra-sim/system/SimRecvCaller.hh"
#include "astra-sim/system/SimSendCaller.hh"
#include "astra-sim/system/StreamBaseline.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include "astra-sim/system/astraccl/CollectiveImpl.hh"
#include "astra-sim/system/astraccl/native_collectives/collective_algorithm/AllToAll.hh"
#include "astra-sim/system/astraccl/native_collectives/collective_algorithm/DoubleBinaryTreeAllReduce.hh"
#include "astra-sim/system/astraccl/native_collectives/collective_algorithm/HalvingDoubling.hh"
#include "astra-sim/system/astraccl/native_collectives/collective_algorithm/Ring.hh"
#include "astra-sim/system/astraccl/native_collectives/logical_topology/BasicLogicalTopology.hh"
#include "astra-sim/system/astraccl/native_collectives/logical_topology/GeneralComplexTopology.hh"
#include "DieModel.hh"
#include "astra-sim/workload/CoarseDieRuntime.hh"
#include "coarse_die/Hardware.hh"
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
            list<BaseStream*>::iterator it;
            this->stream_pointer[base] = it;
            this->queue_id_to_dimension[base] = dimension;
            base++;
        }
        dimension++;
        UsageTracker u(
            2, sys->execution_mode_ != ExecutionDriven::ExecutionMode::Online);
        usage.push_back(u);
    }
}

void Sys::SchedulerUnit::notify_stream_added(int vnet) {
    if (sys->id == 0 &&
        ++total_active_chunks_per_dimension[queue_id_to_dimension[vnet]] == 1) {
        usage[queue_id_to_dimension[vnet]].increase_usage();
    }
    stream_pointer[vnet] = sys->active_Streams[vnet].begin();
    advance(stream_pointer[vnet], running_streams[vnet]);
    while (stream_pointer[vnet] != sys->active_Streams[vnet].end() &&
           running_streams[vnet] < queue_threshold) {
        (*stream_pointer[vnet])->init();
        running_streams[vnet]++;
        advance(stream_pointer[vnet], 1);
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

void Sys::SchedulerUnit::notify_stream_removed(int vnet, Tick running_time) {
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
    stream_pointer[vnet] = sys->active_Streams[vnet].begin();
    advance(stream_pointer[vnet], running_streams[vnet]);
    while (stream_pointer[vnet] != sys->active_Streams[vnet].end() &&
           running_streams[vnet] < queue_threshold) {
        (*stream_pointer[vnet])->init();
        running_streams[vnet]++;
        advance(stream_pointer[vnet], 1);
    }
}
//-----------------------------------------------------------------------------

Sys::Sys(int id,
         string workload_configuration,
         string comm_group_configuration,
         string system_configuration,
         AstraRemoteMemoryAPI* remote_mem,
         AstraNetworkAPI* comm_NI,
         vector<int> physical_dims,
         vector<int> queues_per_dim,
         double injection_scale,
         double comm_scale,
         bool rendezvous_enabled,
         ExecutionDriven::ExecutionMode execution_mode,
         std::shared_ptr<ExecutionDriven::GraphSource> graph_source) {
    this->execution_mode_ = execution_mode;
    this->graph_source_ = std::move(graph_source);

    if ((id + 1) > this->all_sys.size()) {
        this->all_sys.resize(id + 1);
    }
    this->all_sys[id] = this;

    this->id = id;
    this->initialized = false;

    this->workload = nullptr;

    this->roofline_enabled = false;
    this->peak_perf = 0;
    this->roofline = nullptr;
    this->compute_cost_model = ComputeCostModel::LegacyDefault;
    this->die_runtime = nullptr;
    this->coarse_die_runtime = nullptr;
    this->die_model_profile = "";
    this->die_model_config_path = "";

    this->remote_mem = remote_mem;
    this->remote_mem->set_sys(id, this);
    this->local_mem_bw = 0;
    this->local_mem_latency = 0;
    this->remote_mem_bw = 0;
    this->remote_mem_latency = 0;
    this->pipeline_tile_fraction = 0;
    this->hbm_kv_restore_bandwidth_sharing = false;
    this->hbm_bandwidth_contention = true;

    this->memBus = nullptr;
    this->inp_L = 0;
    this->inp_o = 0;
    this->inp_g = 0;
    this->inp_G = 0;
    this->model_shared_bus = 0;
    this->injection_scale = injection_scale;
    this->communication_delay = 0;
    this->local_reduction_delay = 0;

    this->comm_NI = comm_NI;
    this->comm_scale = comm_scale;
    this->rendezvous_enabled = rendezvous_enabled;

    this->scheduler_unit = nullptr;
    this->vLevels = nullptr;
    this->active_chunks_per_dimension = 1;
    this->priority_counter = 0;
    this->pending_events = 0;
    this->preferred_dataset_splits = 0;

    this->first_phase_streams = 0;
    this->total_running_streams = 0;

    this->communication_delay = 10;
    this->local_reduction_delay = 1;

    collective_impl_lookup = new CollectiveImplLookup(id);

    if (initialize_sys(system_configuration) == false) {
        sys_panic("Unable to initialize the system layer because the file can "
                  "not be openned");
    }

    // scheduler
    this->physical_dims = physical_dims;
    this->queues_per_dim = queues_per_dim;
    // The loop below indexes physical_dims by queues_per_dim positions; a
    // shorter physical_dims vector read out of bounds here, and an empty
    // queues_per_dim dereferenced queues_per_dim[0] below. The only caller
    // passes valid equal-length vectors, so this is a pure defensive bound
    // -- fail closed with a named message instead.
    if (queues_per_dim.empty()) {
        sys_panic("queues_per_dim must not be empty");
    }
    for (const int level_count : queues_per_dim) {
        // A zero (or negative) entry would SIGFPE the
        // active_chunks-per-dimension division below and make
        // QueueLevelHandler hand out negative queue ids (the old
        // queue_id *= -1 hot-fix silently flipped them to +1).
        if (level_count < 1) {
            sys_panic("queues_per_dim entries must be >= 1 (got " +
                      std::to_string(level_count) +
                      "; check --num-queues-per-dim / the system "
                      "configuration)");
        }
    }
    if (physical_dims.size() < queues_per_dim.size()) {
        sys_panic("physical_dims (" + std::to_string(physical_dims.size()) +
                  ") is shorter than queues_per_dim (" +
                  std::to_string(queues_per_dim.size()) + ")");
    }
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

    this->concurrent_streams =
        (int)ceil(((double)active_chunks_per_dimension) / queues_per_dim[0]);
    this->active_first_phase = 100000000;
    this->max_running = 100000000;

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
                         graph_source_);
    } else {
        workload =
            new Workload(this, workload_configuration,
                         comm_group_configuration);
    }

    this->initialized = true;
}

Sys::~Sys() {
    if (roofline_enabled) {
        delete this->roofline;
    }

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
        workload = nullptr;
    }

    // Workload teardown may still consult the global simulation clock while
    // DieExecution drops its pending HBM jobs. Keep this Sys reachable until
    // those callbacks, observers, and their model state are fully destroyed;
    // stale outer alarms are separately guarded by handleEvent() once this
    // slot is cleared.
    all_sys[id] = nullptr;

    bool shouldExit = true;
    for (auto& a : all_sys) {
        if (a != nullptr) {
            shouldExit = false;
            break;
        }
    }

    if (shouldExit) {
        exit_sim_loop("Exiting");
    }
}

namespace {

// 3.11 strict latency parse (ported from the wscllm Sys): nlohmann converts
// the JSON value lexically, so a negative literal (-100) reinterpreted as
// uint64_t became ~1.8e19 ns -- downstream arithmetic wrapped the event
// clock (roofline closed form now+runtime) or re-scheduled fluid transfer
// events into the past until the watchdog killed the run, neither of which
// names the field. Fail at the boundary with the field name instead.
uint64_t parse_latency_ns_field(const json& j,
                                const char* key,
                                const string& config_name) {
    const auto& value = j[key];
    if (!value.is_number_integer() && !value.is_number_unsigned()) {
        Sys::sys_panic(std::string(key) +
                       " must be a non-negative integer number of ns in '" +
                       config_name + "'");
    }
    const int64_t parsed = value.get<int64_t>();
    if (parsed < 0 || parsed > (static_cast<int64_t>(1) << 62)) {
        Sys::sys_panic(std::string(key) +
                       " out of domain [0, 2^62] ns (got " +
                       std::to_string(parsed) + " in '" + config_name + "')");
    }
    return static_cast<uint64_t>(parsed);
}

// R7-B port (2026-09-30, from the face repo): strict readers for the
// remaining numeric system.json keys. The implicit nlohmann conversions
// silently truncated fractional values, let negatives produce negative
// event deltas (strict-increase abort far from the key), and accepted
// out-of-domain scheduling constants (active-chunks-per-dimension <= 0
// makes queue_threshold <= 0 so no stream is ever initialized -- a rootless
// lost-wakeup stall). Both readers fail closed naming the key.
double parse_non_negative_number_field(const json& j,
                                       const char* key,
                                       const string& config_name) {
    const auto& value = j[key];
    if (!value.is_number()) {
        Sys::sys_panic(std::string(key) +
                       " must be a number in '" + config_name + "'");
    }
    const double parsed = value.get<double>();
    if (!std::isfinite(parsed) || parsed < 0.0) {
        Sys::sys_panic(std::string(key) +
                       " must be a finite non-negative number in '" +
                       config_name + "' (got " + value.dump() + ")");
    }
    return parsed;
}

int parse_non_negative_int_field(const json& j,
                                 const char* key,
                                 const string& config_name,
                                 int minimum) {
    const auto& value = j[key];
    if (!value.is_number_integer() && !value.is_number_unsigned()) {
        Sys::sys_panic(std::string(key) +
                       " must be an integer in '" + config_name + "'");
    }
    const int64_t parsed = value.get<int64_t>();
    if (parsed < minimum || parsed > std::numeric_limits<int>::max()) {
        Sys::sys_panic(std::string(key) + " out of domain [" +
                       std::to_string(minimum) + ", int-max] in '" +
                       config_name + "' (got " + value.dump() + ")");
    }
    return static_cast<int>(parsed);
}

// Process-level die plan-cache registry (joint rollout stage 1): one shared
// LRU PlanCache per resolved hardware summary. compile() is a pure function
// and a DiePlan is immutable, so every Sys running the same die hardware
// shares its compiled plans instead of recompiling identical
// plan_cache_keys per rank (design doc §8.5: "计划可以缓存"). The
// simulation is single-threaded (one event loop drives all Sys), so the
// registry needs no lock; if a multi-threaded runtime is ever introduced,
// guard this function and the shared DieModel::PlanCache with a mutex.
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

    // Fail-closed configuration parsing (same channel as every other fatal
    // configuration path, same structure as the wscllm Sys): nlohmann
    // throws parse_error for a malformed file and type_error for a field
    // of the wrong JSON type. Neither may escape as an uncaught exception
    // from inside the Sys constructor.
    json j;
    try {
        inFile >> j;
        initialize_sys_fields(j, name);
    } catch (const json::exception& e) {
        sys_panic("system configuration '" + name +
                  "' failed to parse (malformed JSON or a field of the "
                  "wrong JSON type): " +
                  e.what());
    }
    inFile.close();
    // Die backend (joint rollout stage 1, contract §3.2): resolve the
    // compute-cost backend AFTER the whole key-walk so the conflict matrix
    // sees the fully parsed roofline-enabled / replay-only / peak-perf /
    // local-mem-bw state; the die hardware load happens inside for an
    // explicit compute-cost-model=die.
    resolve_compute_cost_backend(j, name);
    return true;
}

void Sys::initialize_sys_fields(const json& j, const string& name) {
    if (j.contains("scheduling-policy")) {
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
    } else {
        // Fail fast instead of scheduling streams with an indeterminate
        // policy: reading an uninitialized enum is undefined behavior, and
        // none of the three legal policies is a safe silent default.
        sys_panic(
            "missing 'scheduling-policy' key in the system configuration "
            "file. The key is required and must be one of \"LIFO\", "
            "\"FIFO\", or \"EXPLICIT\" (the shipped configurations use "
            "\"LIFO\").");
    }
    if (j.contains("collective-optimization")) {
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
        local_reduction_delay = parse_non_negative_int_field(
            j, "local-reduction-delay", name, 0);
    }
    if (j.contains("active-chunks-per-dimension")) {
        active_chunks_per_dimension = parse_non_negative_int_field(
            j, "active-chunks-per-dimension", name, 1);
    }
    if (j.contains("L")) {
        inp_L = parse_non_negative_number_field(j, "L", name);
    }
    if (j.contains("o")) {
        inp_o = parse_non_negative_number_field(j, "o", name);
    }
    if (j.contains("g")) {
        inp_g = parse_non_negative_number_field(j, "g", name);
    }
    if (j.contains("G")) {
        inp_G = parse_non_negative_number_field(j, "G", name);
    }
    if (j.contains("endpoint-delay")) {
        communication_delay = parse_non_negative_int_field(
            j, "endpoint-delay", name, 0);
        communication_delay = communication_delay * injection_scale;
    }
    if (j.contains("model-shared-bus")) {
        int inp_model_shared_bus = j["model-shared-bus"];
        if (inp_model_shared_bus == 1) {
            model_shared_bus = true;
        } else {
            model_shared_bus = false;
        }
    } else {
        model_shared_bus = false;
    }
    if (j.contains("preferred-dataset-splits")) {
        preferred_dataset_splits = j["preferred-dataset-splits"];
    }
    if (j.contains("peak-perf")) {
        peak_perf = j["peak-perf"];
        peak_perf = peak_perf * 1000000000000;  // TFLOPS
    }
    if (j.contains("local-mem-bw")) {
        local_mem_bw = j["local-mem-bw"];
        local_mem_bw = local_mem_bw * 1000000000;  // GB/sec
    }
    if (j.contains("local-mem-latency")) {
        // Strict latency parse (3.11, ported from the wscllm Sys): nlohmann
        // converts the JSON value lexically, so a negative literal (-100)
        // reinterpreted as uint64_t became ~1.8e19 ns -- the roofline
        // closed-form path computed now+runtime with it, wrapped the event
        // clock, and tripped the EventQueue strict-increase abort (whose
        // message does not name this field); on the fluid HBM path the
        // same value re-scheduled transfer events into the past until the
        // watchdog killed the run. Fail here, at the boundary, naming the
        // field.
        local_mem_latency = parse_latency_ns_field(j, "local-mem-latency",
                                                   name);
    }
    if (j.contains("remote-mem-bw")) {
        remote_mem_bw = j["remote-mem-bw"];
        remote_mem_bw = remote_mem_bw * 1000000000;  // GB/sec
    }
    if (j.contains("remote-mem-latency")) {
        // Same wraparound exposure as local-mem-latency above.
        remote_mem_latency = parse_latency_ns_field(j, "remote-mem-latency",
                                                    name);
    }
    if (j.contains("pipeline-tile-fraction")) {
        pipeline_tile_fraction = j["pipeline-tile-fraction"];
        pipeline_tile_fraction =
            std::max(0.0, std::min(1.0, pipeline_tile_fraction));
    }
    if (j.contains("hbm-kv-restore-bandwidth-sharing")) {
        const auto& sharing = j["hbm-kv-restore-bandwidth-sharing"];
        if (sharing.is_boolean()) {
            hbm_kv_restore_bandwidth_sharing = sharing.get<bool>();
        } else if (sharing.is_number_integer() || sharing.is_number_unsigned()) {
            hbm_kv_restore_bandwidth_sharing = sharing.get<int64_t>() != 0;
        } else {
            sys_panic(
                "hbm-kv-restore-bandwidth-sharing must be boolean or integer");
        }
    }
    // N-way equal-split HBM contention (code default true). true = the
    // LocalHbmBandwidthModel owns every HBM user (COMP / restore / NoC comm
    // endpoints / pool endpoints); false = the legacy behavior is fully
    // preserved (the old flag above keeps its two-user 50/50 semantics for
    // A/B). local-mem-bw <= 0 auto-disables: without a positive bus there is
    // nothing to split.
    if (j.contains("hbm-bandwidth-contention")) {
        const auto& contention = j["hbm-bandwidth-contention"];
        if (contention.is_boolean()) {
            hbm_bandwidth_contention = contention.get<bool>();
        } else if (contention.is_number_integer() ||
                   contention.is_number_unsigned()) {
            hbm_bandwidth_contention = contention.get<int64_t>() != 0;
        } else {
            sys_panic(
                "hbm-bandwidth-contention must be boolean or integer");
        }
    }
    // coarse_die_v2 derives its HBM rows after this key walk. Preserve
    // the user flag until its descriptor is loaded and validated.
    if (local_mem_bw <= 0 &&
        !(j.contains("die-model-profile") &&
          j["die-model-profile"] == "coarse_die_v2")) {
        hbm_bandwidth_contention = false;
    }
    if (j.contains("roofline-enabled")) {
        // Boolean/integer strict parsing (same fix as the wscllm Sys):
        // nlohmann's cross-type `!= 0` treated a JSON `false` as enabled
        // (fail-open).
        const auto& roofline_flag = j["roofline-enabled"];
        if (roofline_flag.is_boolean()) {
            roofline_enabled = roofline_flag.get<bool>();
        } else if (roofline_flag.is_number_integer() ||
                   roofline_flag.is_number_unsigned()) {
            roofline_enabled = roofline_flag.get<int64_t>() != 0;
        } else {
            sys_panic("roofline-enabled must be boolean or integer");
        }
        if (roofline_enabled) {
            roofline = new Roofline(local_mem_bw, peak_perf);
        }
    }
    this->trace_enabled = false;
    if (j.contains("trace-enabled")) {
        const auto& trace_flag = j["trace-enabled"];
        if (trace_flag.is_boolean()) {
            this->trace_enabled = trace_flag.get<bool>();
        } else if (trace_flag.is_number_integer() ||
                   trace_flag.is_number_unsigned()) {
            this->trace_enabled = trace_flag.get<int64_t>() != 0;
        } else {
            sys_panic("trace-enabled must be boolean or integer");
        }
    }
    this->replay_only = false;
    if (j.contains("replay-only")) {
        const auto& replay_flag = j["replay-only"];
        if (replay_flag.is_boolean()) {
            this->replay_only = replay_flag.get<bool>();
        } else if (replay_flag.is_number_integer() ||
                   replay_flag.is_number_unsigned()) {
            this->replay_only = replay_flag.get<int64_t>() != 0;
        } else {
            sys_panic("replay-only must be boolean or integer");
        }
    }
    this->track_local_mem = false;
    if (j.contains("track-local-mem")) {
        const auto& track_flag = j["track-local-mem"];
        if (track_flag.is_boolean()) {
            this->track_local_mem = track_flag.get<bool>();
        } else if (track_flag.is_number_integer() ||
                   track_flag.is_number_unsigned()) {
            this->track_local_mem = track_flag.get<int64_t>() != 0;
        } else {
            sys_panic("track-local-mem must be boolean or integer");
        }
    }

    this->local_mem_trace_filename = "local_mem_trace";
    if (j.contains("local-mem-trace-filename")) {
        this->local_mem_trace_filename = j["local-mem-trace-filename"];
    }

    // Unknown-key gate (3.12, 2026-09-29, same as the wscllm Sys): this
    // reader only consumed known keys and silently ignored everything
    // else, so a typo'd template key fell back to the built-in default and
    // the feature silently turned off with zero diagnostics -- fail-open
    // at the last line of defense. The hardware json side already enforces
    // an exact-key whitelist (_require_exact_keys in the resolver); this
    // is the matching gate for the system side. The retired
    // *-implementation-custom keys are listed as known so they keep their
    // specific reject_retired_custom_key diagnosis inside
    // setup_collective_impl_from_config (called right after this gate)
    // instead of the generic unknown-key message. Keep this set in sync
    // with _KNOWN_SYSTEM_TEMPLATE_KEYS in sh_test_mesh/config_resolver.py.
    static const std::set<std::string> known_system_keys = {
        "scheduling-policy",
        "collective-optimization",
        "local-reduction-delay",
        "active-chunks-per-dimension",
        "L",
        "o",
        "g",
        "G",
        "endpoint-delay",
        "model-shared-bus",
        "preferred-dataset-splits",
        "peak-perf",
        "local-mem-bw",
        "local-mem-latency",
        "remote-mem-bw",
        "remote-mem-latency",
        "pipeline-tile-fraction",
        "hbm-kv-restore-bandwidth-sharing",
        "hbm-bandwidth-contention",
        "roofline-enabled",
        "trace-enabled",
        "replay-only",
        "track-local-mem",
        "local-mem-trace-filename",
        "compute-cost-model",
        "die-model-profile",
        "die-model-config",
        "all-reduce-implementation",
        "reduce-scatter-implementation",
        "all-gather-implementation",
        "all-to-all-implementation",
        // retired keys: known, but rejected with their own message below.
        "all-reduce-implementation-custom",
        "all-gather-implementation-custom",
        "reduce-scatter-implementation-custom",
        "all-to-all-implementation-custom",
        "per-node-custom-implementation",
    };
    std::string unknown_keys;
    for (auto it = j.begin(); it != j.end(); ++it) {
        if (known_system_keys.count(it.key()) == 0) {
            if (!unknown_keys.empty()) {
                unknown_keys += ", ";
            }
            unknown_keys += "\"" + it.key() + "\"";
        }
    }
    if (!unknown_keys.empty()) {
        sys_panic("system configuration '" + name +
                  "' contains unknown key(s): " + unknown_keys +
                  " (a typo'd key used to silently fall back to the default "
                  "value and disable the intended feature)");
    }
    collective_impl_lookup->setup_collective_impl_from_config(j);
}

void Sys::resolve_compute_cost_backend(const json& j, const string& name) {
    // Compute cost backend resolution (joint rollout stage 1; contract §3.2
    // conflict matrix, fail-closed on every branch -- field parse order must
    // never select a backend silently). Runs after the whole key-walk.
    const bool explicit_present = j.contains("compute-cost-model");
    ComputeCostModel explicit_model = ComputeCostModel::LegacyDefault;
    if (explicit_present) {
        if (!j["compute-cost-model"].is_string()) {
            sys_panic(
                "system config key 'compute-cost-model' must be a string");
        }
        const string model = j["compute-cost-model"];
        if (model == "die") {
            explicit_model = ComputeCostModel::Die;
        } else if (model == "roofline") {
            explicit_model = ComputeCostModel::Roofline;
        } else if (model == "replay") {
            explicit_model = ComputeCostModel::Replay;
        } else {
            sys_panic("unknown value for compute-cost-model in sys input "
                      "file: '" +
                      model +
                      "' (legal: die | roofline | replay)");
        }
    }
    const bool profile_present = j.contains("die-model-profile");
    string profile;
    if (profile_present) {
        if (!j["die-model-profile"].is_string()) {
            sys_panic(
                "system config key 'die-model-profile' must be a string");
        }
        profile = j["die-model-profile"];
        this->die_model_profile = profile;
        if (profile != "mesh_die_v1" && profile != "tx8_reference_v1" &&
            profile != "coarse_die_v2") {
            sys_panic("unknown value for die-model-profile in sys input "
                      "file: '" +
                      profile + "' (legal: mesh_die_v1 | tx8_reference_v1 | coarse_die_v2)");
        }
    }
    const bool config_explicit_present = j.contains("die-model-config");
    string config_path;
    if (config_explicit_present) {
        if (!j["die-model-config"].is_string()) {
            sys_panic(
                "system config key 'die-model-config' must be a string");
        }
        config_path = j["die-model-config"];
    }

    if (explicit_present && explicit_model == ComputeCostModel::Die) {
        if (roofline_enabled) {
            sys_panic("system config conflict: compute-cost-model=die is "
                      "incompatible with roofline-enabled=1 (contract 3.2)");
        }
        if (replay_only) {
            sys_panic("system config conflict: compute-cost-model=die is "
                      "incompatible with replay-only=1 (contract 3.2)");
        }
        initialize_die_backend(name, config_explicit_present, config_path,
                               profile_present, profile,
                               j.contains("peak-perf"),
                               j.contains("local-mem-bw"),
                               j.contains("hbm-bandwidth-contention"));
    } else if (explicit_present &&
               explicit_model == ComputeCostModel::Roofline) {
        // Explicit legacy roofline backend: same dispatch state as the
        // historical roofline-enabled=1 configuration.
        compute_cost_model = ComputeCostModel::Roofline;
        if (!roofline_enabled) {
            roofline_enabled = true;
            roofline = new Roofline(local_mem_bw, peak_perf);
        }
    } else if (explicit_present) {
        // Explicit replay backend: the legacy no-roofline dispatch
        // (COMP nodes run through issue_replay semantics).
        compute_cost_model = ComputeCostModel::Replay;
        if (profile_present || config_explicit_present) {
            LoggerFactory::get_logger("system")->warn(
                "system config file {} carries die-model keys but "
                "compute-cost-model is not 'die'; they are ignored",
                name);
        }
    } else {
        // No compute-cost-model key: backward-compatible dispatch. The die
        // keys never select the die backend on their own.
        compute_cost_model = roofline_enabled
            ? ComputeCostModel::Roofline
            : ComputeCostModel::LegacyDefault;
        if (profile_present || config_explicit_present) {
            LoggerFactory::get_logger("system")->warn(
                "system config file {} carries die-model keys but no "
                "compute-cost-model=die; they are ignored",
                name);
        }
    }
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

    // coarse_die_v2 is a separate typed backend. COMP plans and all local
    // HBM endpoint traffic are submitted to the same per-rank Runtime.
    if (profile == "coarse_die_v2") {
        if (!profile_present || die_cfg.value("schema", string()) !=
                                    "coarse_die_v2") {
            sys_panic(
                "coarse_die_v2 requires an explicit die-model-profile and a "
                "coarse_die_v2 resolved descriptor at " + path);
        }
        if (hbm_contention_key_present && !hbm_bandwidth_contention) {
            sys_panic(
                "coarse_die_v2 requires shared local HBM service for COMP "
                "and endpoint traffic; hbm-bandwidth-contention=0 is not "
                "supported");
        }
        coarse_die::Hardware coarse_hw;
        try {
            coarse_hw = coarse_die::resolve_hardware(die_cfg);
        } catch (const std::exception& e) {
            LoggerFactory::get_logger("system")->critical(
                "coarse_die_v2 descriptor {} rejected: {}", path, e.what());
            exit(1);
        }
        coarse_die_runtime = std::make_unique<CoarseDieRuntime>(
            this, std::move(coarse_hw));
        die_runtime.reset();
        die_model_profile = "coarse_die_v2";
        if (!local_mem_bw_key_present) {
            local_mem_bw = coarse_die_runtime->hardware().rates[
                               coarse_die::HBM0] +
                           coarse_die_runtime->hardware().rates[
                               coarse_die::HBM1];
        }
        hbm_bandwidth_contention = true;
        return;
    }
    if (profile_present && profile != "mesh_die_v1" &&
        profile != "tx8_reference_v1") {
        sys_panic("legacy DieModel cannot load die-model-profile '" + profile +
                  "'");
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
    if (profile_present && profile != DieModel::to_string(hw.profile)) {
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
        // (This repository defaults hbm_bandwidth_contention to true in the
        // constructor; the auto-disable only fired on the unfilled zero.)
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

void Sys::sys_panic(string msg) {
    auto logger = LoggerFactory::get_logger("system");
    logger->critical(msg);
    exit(1);
}

void Sys::exit_sim_loop(string msg) {
    auto logger = LoggerFactory::get_logger("system");
    logger->warn(msg);
}

void Sys::call(EventType type, CallData* data) {}

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
            pending_events--;
            callable.callable->call(callable.event, callable.call_data);
        } catch (const std::exception& e) {
            auto logger = LoggerFactory::get_logger("system");
            logger->critical("warning! a callable is removed before call {}",
                             e.what());
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
    // Fail-closed on delta==0 (latent-trap guard, same as the wscllm Sys):
    // the outer alarm goes through the backend's cancellable schedule,
    // which clamps a delta==0 request to +1 ns OUTSIDE invoke context and
    // merges at current_time INSIDE it -- two different instants -- while
    // this Sys-side bucket is keyed at boostedTick()+delta unconditionally.
    // A delta==0 caller therefore got an alarm and an event bucket on two
    // different calendars (orphaned events, silently dropped wakeups)
    // depending on which context the call ran in. The only caller clamps
    // to >= 1 already, so the guard costs nothing today and turns the
    // ambiguous request into a named error instead.
    if (delta_cycles == 0) {
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
    pending_events++;
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
        pending_events--;
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
    // H3 (face-LRU audit, 2026-09-29, mirrored to joint): a delta==0
    // registration outside the backend's invoke pass hits the EventQueue's
    // same-tick guard and exits the whole run. Online tick-end / deferred
    // commits issue collectives on this path, and a communication_delay
    // truncated to 0 by a fractional injection scale (legal CLI combo, e.g.
    // endpoint-delay 10 with --injection-scale 0.05) makes the first
    // collective packet fatal with a message unrelated to its cause.
    // Same-tick merging stays byte-exact inside the invoke pass (static
    // dispatch), so only the forbidden context defers to the next tick.
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
    pending_events++;
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
        // (The NPU_to_MA / MA_to_NPU arm that used to live here had no
        // producer: MemBus routes through register_event's CallEvents
        // channel, and unlike the arm above it lacked both the delete of
        // the handler data and the all_sys bounds guard -- removed with the
        // 2026-10-01 dead-code audit.)
    } else if (event == EventType::RendezvousSend) {
        RendezvousSendData* rsd = (RendezvousSendData*)ehd;
        rsd->send.call(EventType::General, nullptr);
        delete rsd;
    } else if (event == EventType::RendezvousRecv) {
        RendezvousRecvData* rrd = (RendezvousRecvData*)ehd;
        rrd->recv.call(EventType::General, nullptr);
        delete rrd;
    } else if ((event == EventType::CompFinished) ||
               (event == EventType::MemLoadFinished) ||
               (event == EventType::MemStoreFinished)) {
        MemEventHandlerData* mehd = (MemEventHandlerData*)ehd;
        if (mehd->workload) {
            mehd->workload->call(event, mehd->wlhd);
        }
        delete mehd;
    } else if (event == EventType::PacketReceived) {
        RecvPacketEventHandlerData* rcehd = (RecvPacketEventHandlerData*)ehd;
        if (rcehd->workload) {
            rcehd->workload->call(event, rcehd->wlhd);
        } else if (rcehd->callable) {
            rcehd->callable->call(event, rcehd->wlhd);
        }
        if (rcehd->owner) {
            rcehd->owner->consume(rcehd);
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
        sys_panic("no known logical topology!");
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
        // (2026-10-01 R3 审查修复) 与 CommunicatorGroup 的 M2 fail-
        // closed 纪律对称：裸 throw 会被 call_events 的 catch 吞
        // 成一条 critical，发起节点已 take/occupy 却永不完成
        // （run 静默不完整收场）。缺 *-implementation 键的
        // system.json 在此点名退出。
        try {
            implementation_per_dimension =
                collective_impl_lookup->get_collective_impl(
                    ComType::All_Reduce, workload_node_id);
        } catch (const std::exception& e) {
            LoggerFactory::get_logger("system::Sys")
                ->critical(
                    "######### Exiting because the collective "
                    "implementation lookup failed (com_type=", 
                    static_cast<int>(ComType::All_Reduce), 
                    ", workload_node_id=", workload_node_id, 
                    ", communicator_group=<null> -- check the "
                    "*-implementation keys in the system "
                    "configuration): ", e.what(), " #########");
            std::exit(1);
        }
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
        // (2026-10-01 R3 审查修复) 与 CommunicatorGroup 的 M2 fail-
        // closed 纪律对称：裸 throw 会被 call_events 的 catch 吞
        // 成一条 critical，发起节点已 take/occupy 却永不完成
        // （run 静默不完整收场）。缺 *-implementation 键的
        // system.json 在此点名退出。
        try {
            implementation_per_dimension =
                collective_impl_lookup->get_collective_impl(
                    ComType::All_to_All, workload_node_id);
        } catch (const std::exception& e) {
            LoggerFactory::get_logger("system::Sys")
                ->critical(
                    "######### Exiting because the collective "
                    "implementation lookup failed (com_type=", 
                    static_cast<int>(ComType::All_to_All), 
                    ", workload_node_id=", workload_node_id, 
                    ", communicator_group=<null> -- check the "
                    "*-implementation keys in the system "
                    "configuration): ", e.what(), " #########");
            std::exit(1);
        }
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
        // (2026-10-01 R3 审查修复) 与 CommunicatorGroup 的 M2 fail-
        // closed 纪律对称：裸 throw 会被 call_events 的 catch 吞
        // 成一条 critical，发起节点已 take/occupy 却永不完成
        // （run 静默不完整收场）。缺 *-implementation 键的
        // system.json 在此点名退出。
        try {
            implementation_per_dimension =
                collective_impl_lookup->get_collective_impl(
                    ComType::All_Gather, workload_node_id);
        } catch (const std::exception& e) {
            LoggerFactory::get_logger("system::Sys")
                ->critical(
                    "######### Exiting because the collective "
                    "implementation lookup failed (com_type=", 
                    static_cast<int>(ComType::All_Gather), 
                    ", workload_node_id=", workload_node_id, 
                    ", communicator_group=<null> -- check the "
                    "*-implementation keys in the system "
                    "configuration): ", e.what(), " #########");
            std::exit(1);
        }
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
        // (2026-10-01 R3 审查修复) 与 CommunicatorGroup 的 M2 fail-
        // closed 纪律对称：裸 throw 会被 call_events 的 catch 吞
        // 成一条 critical，发起节点已 take/occupy 却永不完成
        // （run 静默不完整收场）。缺 *-implementation 键的
        // system.json 在此点名退出。
        try {
            implementation_per_dimension =
                collective_impl_lookup->get_collective_impl(
                    ComType::Reduce_Scatter, workload_node_id);
        } catch (const std::exception& e) {
            LoggerFactory::get_logger("system::Sys")
                ->critical(
                    "######### Exiting because the collective "
                    "implementation lookup failed (com_type=", 
                    static_cast<int>(ComType::Reduce_Scatter), 
                    ", workload_node_id=", workload_node_id, 
                    ", communicator_group=<null> -- check the "
                    "*-implementation keys in the system "
                    "configuration): ", e.what(), " #########");
            std::exit(1);
        }
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
    // TODO(jinsun): For custom collective, we do not need the chunk_size here (since the chunk size is already determined)
    // Therefore, we also do not need the 'preferred-dataset-splits' value from the system JSON input.
    // However, this variable is intertwined deeply in this function so that we cannot remove it for now.
    // Therefore, we have to keep that value in the JSON input. TODO: Refactor and remove.
    // H2 (face-LRU audit, 2026-09-29, mirrored to joint): everything below
    // indexes dimensions_involved with dim_mapper[dim] for dim < topology's
    // logical dimension count. The legacy ET default is a 4x-true list
    // (NodeStore), which reads fine only while the topology has at most 4
    // dimensions -- a 5+-dimensional collective (or a hand-written short
    // list) would otherwise index vector<bool> past the end, and the garbage
    // bit read decides whether a dimension is silently skipped. A longer
    // list stays tolerated (byte-exact with the legacy default); only a
    // short one is fatal.
    const int logical_dims = topology->get_num_of_dimensions();
    if (static_cast<int>(dimensions_involved.size()) < logical_dims) {
        LoggerFactory::get_logger("system::Sys")
            ->critical(
                "######### Exiting because dimensions_involved has {} "
                "entries but the collective topology has {} logical "
                "dimensions (sys.id={}) #########",
                dimensions_involved.size(), logical_dims, id);
        std::exit(1);
    }
    // A zero-size collective cannot be split into streams: ceil(0/chunk) = 0
    // streams would leave the DataSet unfinished forever (the notifier
    // never fires) and the non-All_Gather path in determine_chunk_size
    // would inflate size back up to splits * total_nodes and time the
    // empty collective as real traffic. Fail closed instead (2026-10-01
    // audit, same as the wscllm Sys).
    if (size == 0) {
        sys_panic("collective data size must be positive");
    }
    uint64_t chunk_size = determine_chunk_size(size, collective_type);
    if (chunk_size == 0) {
        sys_panic("collective chunk size must be positive");
    }
    int streams = ceil(((double)size) / chunk_size);
    uint64_t remain_size;
    DataSet* dataset = new DataSet(streams);
    int pri = get_priority(explicit_priority);
    int count = 0;

    // H4 (R7-A) defense in depth (ported from the face repo, 2026-09-30):
    // every dim loop below indexes the queue levels (built from the
    // PHYSICAL dimension count) with this topology's dimension ids. A
    // topology with more dimensions than physical queue levels (e.g. a
    // hand-written shaped comm group) used to read those tables out of
    // bounds. The group constructor rejects the same shape at load time and
    // the dimensions_involved length is checked by the earlier H2 gate
    // above; this is the fail-loud backstop at the point of use.
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

        if (chunk_size > size) {
            size = 0;
        } else {
            size -= chunk_size;
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
                    InjectionPolicy::Normal,
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
                    InjectionPolicy::Normal,
                    implementation_per_dimension[dim_mapper[dim]]);
                vect.push_back(phase);
                remain_size = phase.final_data_size;
            }
            while (dim > 0 && (dimensions_involved[dim_mapper[dim]] == false ||
                               topology->get_num_of_nodes_in_dimension(
                                   dim_mapper[dim]) == 1)) {
                dim--;
            }

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
                    InjectionPolicy::Normal,
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
                    InjectionPolicy::Normal,
                    implementation_per_dimension[dim_mapper[dim]]);
                vect.push_back(phase);
                remain_size = phase.final_data_size;
            }
        }
        if (vect.size() > 0) {
            int stream_id = num_streams++;
            if (communicator_group != nullptr) {
                stream_id = communicator_group->num_streams++;
            }
            StreamBaseline* newStream =
                new StreamBaseline(this, dataset, stream_id, vect, pri);
            newStream->current_queue_id = -1;
            insert_into_ready_list(newStream);
        } else {
            // Zero-phase collective (every dimension is a singleton or not
            // involved -- e.g. a 1-rank communicator group): no stream is
            // ever created, so the dataset's notifier would never fire and
            // the requesting workload node would hang forever with zero
            // diagnostics (2026-10-01 audit: dataset->active has no reader
            // anywhere; the deactivation path was a silent hang). Complete
            // the dataset immediately instead -- nothing to communicate.
            dataset->active = false;
            dataset->total_streams = 0;
            dataset->notify_stream_finished(nullptr);
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
    InjectionPolicy injection_policy,
    CollectiveImpl* collective_impl,
    CommunicatorGroup* comm_group) {
    if (collective_impl->type == CollectiveImplType::Ring ||
        collective_impl->type == CollectiveImplType::OneRing) {
        CollectivePhase vn(this, queue_id,
                           new Ring(collective_type, id,
                                    (RingTopology*)topology, data_size,
                                    direction, injection_policy));
        return vn;
    } else if (collective_impl->type == CollectiveImplType::Direct ||
               collective_impl->type == CollectiveImplType::OneDirect) {
        CollectivePhase vn(this, queue_id,
                           new AllToAll(collective_type,
                                        ((DirectCollectiveImpl*)collective_impl)
                                            ->direct_collective_window,
                                        id, (RingTopology*)topology, data_size,
                                        direction, InjectionPolicy::Normal));
        return vn;
    } else if (collective_impl->type == CollectiveImplType::DoubleBinaryTree) {
        CollectivePhase vn(this, queue_id,
                           new DoubleBinaryTreeAllReduce(
                               id, (BinaryTree*)topology, data_size));
        return vn;
    } else if (collective_impl->type == CollectiveImplType::HalvingDoubling ||
               collective_impl->type ==
                   CollectiveImplType::OneHalvingDoubling) {
        CollectivePhase vn(this, queue_id,
                           new HalvingDoubling(collective_type, id,
                                               (RingTopology*)topology,
                                               data_size));
        return vn;
    } else {
        LoggerFactory::get_logger("system")->critical(
            "Error: No known collective implementation for collective phase");
        exit(1);
    }
}

uint64_t Sys::determine_chunk_size(uint64_t& size, ComType type) {
    if (preferred_dataset_splits <= 0) {
        // Fail fast instead of dividing by zero (SIGFPE): the constructor
        // default is 0 and the JSON input only overrides it when the key is
        // present, so a missing key or a non-positive value lands here on
        // the first collective communication.
        sys_panic(
            "invalid or missing 'preferred-dataset-splits' key in the system "
            "configuration file. The key is required by any workload that "
            "issues collective communications and must be a positive integer "
            "giving the number of streams a collective is split into (the "
            "shipped configurations use 6).");
    }
    uint64_t chunk_size = size / preferred_dataset_splits;
    if (type == ComType::All_Gather && chunk_size == 0) {
        // An All_Gather payload smaller than 'preferred-dataset-splits'
        // divides into zero-size chunks; ceil(size/0) would then overflow
        // the stream count (UB) and the emit loop would never decrement
        // size. Fall back to one chunk covering the whole size (2026-10-01
        // audit, same as the wscllm Sys).
        return size;
    }
    // We want the collective size to have minimum size, otherwise, there is a
    // possibility of size overflow due to further dividing it to more
    // fine-grained messages
    if (type != ComType::All_Gather && this->total_nodes > chunk_size) {
        chunk_size = this->total_nodes;
        size = preferred_dataset_splits * chunk_size;
    }
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

    // should not reach here
    assert(false);
    std::exit(-1);
}

void Sys::insert_into_ready_list(BaseStream* stream) {
    insert_stream(&ready_list, stream);
    scheduler_unit->notify_stream_added_into_ready_list();
}

void Sys::insert_stream(list<BaseStream*>* queue, BaseStream* baseStream) {
    // Intra-dimension scheduling is fixed to FIFO: skip streams that are
    // already initialized and streams whose priority is not smaller than the
    // new stream's priority.
    list<BaseStream*>::iterator it = queue->begin();
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
        // R1-D-L4 port (2026-09-30): a null slot (torn-down rank) makes the
        // all-rank front scan unevaluable; the all-or-nothing contract
        // returns instead of dereferencing it.
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
        int top_vn = ready_list.front()->phases_to_go.front().queue_id;
        int total_waiting_streams = ready_list.size();
        int total_phases = ready_list.front()->phases_to_go.size();

        proceed_to_next_vnet_baseline((StreamBaseline*)ready_list.front());

        if (ready_list.front()->current_queue_id == -1) {
            Sys::sys_panic("should not happen! top queue id: " +
                           to_string(top_vn) + " , total phases: " +
                           to_string(total_phases) +
                           " , waiting streams: " +
                           to_string(total_waiting_streams));
        }

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
    // The write-only per-stream network-statistic accumulation was removed
    // (face-LRU audit 2026-09-29 mirrored to joint); nothing ever read it.
    if (stream->my_current_phase.algorithm != nullptr) {
        delete stream->my_current_phase.algorithm;
    }
    if (stream->phases_to_go.size() == 0) {
        stream->dataset->notify_stream_finished((StreamStat*)stream);
    }
    if (stream->current_queue_id >= 0 && stream->my_current_phase.enabled) {
        list<BaseStream*>& target =
            active_Streams.at(stream->my_current_phase.queue_id);
        for (list<BaseStream*>::iterator it = target.begin();
             it != target.end(); ++it) {
            if (((StreamBaseline*)(*it))->stream_id == stream->stream_id) {
                target.erase(it);
                break;
            }
        }
    }
    if (stream->phases_to_go.size() == 0) {
        total_running_streams--;
        if (previous_vnet >= 0) {
            scheduler_unit->notify_stream_removed(
                previous_vnet, Sys::boostedTick() - stream->last_init);
        }
        delete stream;
        return;
    }
    stream->steps_finished++;
    // This is hot fix for random failures of simulation.
    // TODO: Need to find a better way for negative queue id occurrence.
    if (stream->phases_to_go.front().queue_id < 0) {
        stream->phases_to_go.front().queue_id *= -1;
    }
    stream->current_queue_id = stream->phases_to_go.front().queue_id;
    stream->current_com_type = stream->phases_to_go.front().comm_type;

    CollectivePhase vi = stream->phases_to_go.front();
    stream->my_current_phase = vi;
    stream->phases_to_go.pop_front();
    stream->initialized = false;
    stream->total_packets_sent = 0;

    // (net_message accumulator init removed with the dead statistic chain.)

    if (stream->my_current_phase.enabled) {
        insert_stream(&active_Streams[stream->current_queue_id], stream);
    }

    stream->state = StreamState::Ready;

    if (previous_vnet >= 0) {
        scheduler_unit->notify_stream_removed(
            previous_vnet, Sys::boostedTick() - stream->last_init);
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

int Sys::front_end_sim_send_correlated(
    Tick delay,
    void* buffer,
    uint64_t count,
    int type,
    int dst,
    int tag,
    sim_request* request,
    Sys::FrontEndSendRecvType send_type,
    void (*msg_handler)(void* fun_arg),
    void* fun_arg,
    const AstraNetworkAPI::LogicalFlowKey& key) {
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
                                   request, msg_handler, fun_arg, key);
    }
    return sim_send_correlated(delay, buffer, count, type, dst, tag, request,
                               msg_handler, fun_arg, key);
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
                             void* fun_arg,
                             std::optional<AstraNetworkAPI::LogicalFlowKey>
                                 logical_key) {
    if (tag >= Sys::FrontEndSendRecvType::RENDEZVOUS) {
        sys_panic("tag is bigger than RENDEZVOUS_COMM_TAG_OFFSET, \
        which means it might be mistakenly used as a rendezvous tag.");
    }
    RendezvousSendData* rsd =
        new RendezvousSendData(id, this, buffer, count, type, dst, tag,
                               *request, msg_handler, fun_arg,
                               std::move(logical_key));
    sim_request newReq = *request;
    uint64_t rendevouz_size = 8192;
    newReq.dstRank = request->srcRank;
    newReq.srcRank = request->dstRank;
    newReq.reqCount = rendevouz_size;
    int newTag = tag + Sys::FrontEndSendRecvType::RENDEZVOUS;
    newReq.tag = newTag;
    // M2 fix (ported from the face repo, 2026-09-30): the handshake is a
    // fixed 8192-byte control transfer that used to point straight at the
    // caller's message buffer, so any backend that really delivers into a
    // receive buffer would overflow a message smaller than 8192 bytes.
    // Neither shipped backend dereferences handshake bytes (control-only
    // transfer, single-threaded event loop); do not resize the user buffer
    // to 8192 -- the handshake size is part of the modeled timing.
    static uint8_t handshake_buffer[8192];
    sim_recv(delay, handshake_buffer, rendevouz_size, type, dst, newTag,
             &newReq, &Sys::handleEvent, rsd);
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
    newReq.reqCount = rendevouz_size;
    int newTag = tag + Sys::FrontEndSendRecvType::RENDEZVOUS;
    newReq.tag = newTag;
    // M2 fix: see rendezvous_sim_send (shared handshake buffer, not the
    // caller's payload buffer).
    static uint8_t handshake_buffer_recv[8192];
    sim_send(delay, handshake_buffer_recv, rendevouz_size, type, src, newTag,
             &newReq, &Sys::handleEvent, rrd);
    return 1;
}

int Sys::sim_send(Tick delay,
                  void* buffer,
                  uint64_t count,
                  int type,
                  int dst,
                  int tag,
                  sim_request* request,
                  void (*msg_handler)(void* fun_arg),
                  void* fun_arg) {
    if (delay == 0) {
        comm_NI->sim_send(buffer, count, type, dst, tag, request, msg_handler,
                          fun_arg);
    } else {
        try_register_event(new SimSendCaller(this, buffer, count, type, dst,
                                             tag, *request, msg_handler,
                                             fun_arg, true),
                           EventType::General, nullptr, delay);
    }
    return 1;
}

int Sys::sim_send_correlated(
    Tick delay,
    void* buffer,
    uint64_t count,
    int type,
    int dst,
    int tag,
    sim_request* request,
    void (*msg_handler)(void* fun_arg),
    void* fun_arg,
    const AstraNetworkAPI::LogicalFlowKey& key) {
    if (delay == 0) {
        comm_NI->sim_send_correlated(buffer, count, type, dst, tag, request,
                                     msg_handler, fun_arg, key);
    } else {
        try_register_event(
            new SimSendCaller(this, buffer, count, type, dst, tag, *request,
                              msg_handler, fun_arg, true, key),
            EventType::General, nullptr, delay);
    }
    return 1;
}

int Sys::sim_recv(Tick delay,
                  void* buffer,
                  uint64_t count,
                  int type,
                  int src,
                  int tag,
                  sim_request* request,
                  void (*msg_handler)(void* fun_arg),
                  void* fun_arg) {
    if (delay == 0) {
        comm_NI->sim_recv(buffer, count, type, src, tag, request, msg_handler,
                          fun_arg);
    } else {
        try_register_event(new SimRecvCaller(this, buffer, count, type, src,
                                             tag, *request, msg_handler,
                                             fun_arg, true),
                           EventType::General, nullptr, delay);
    }
    return 1;
}
}  // namespace AstraSim
