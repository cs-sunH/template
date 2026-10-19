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
#include <stdexcept>
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
#include "astra-sim/system/astraccl/native_collectives/collective_algorithm/AllToAll.hh"
#include "astra-sim/system/astraccl/native_collectives/collective_algorithm/DoubleBinaryTreeAllReduce.hh"
#include "astra-sim/system/astraccl/native_collectives/collective_algorithm/HalvingDoubling.hh"
#include "astra-sim/system/astraccl/native_collectives/collective_algorithm/Ring.hh"
#include "astra-sim/system/scheduling/OfflineGreedy.hh"
#include "astra-sim/system/astraccl/native_collectives/logical_topology/BasicLogicalTopology.hh"
#include "astra-sim/system/astraccl/native_collectives/logical_topology/GeneralComplexTopology.hh"
#include "astra-sim/workload/CoarseDieRuntime.hh"
#include "astra-sim/workload/WorkspaceAdmission.hh"
#include "DieModel.hh"
#include "coarse_die/Hardware.hh"
#include <json/json.hpp>

using namespace std;
using json = nlohmann::json;

namespace AstraSim {
uint8_t* Sys::dummy_data = new uint8_t[2];
vector<Sys*> Sys::all_sys;

namespace {

void cancel_call_events_alarm(void* const arg) {
    delete static_cast<BasicEventHandlerData*>(arg);
}

// Process-level die plan-cache registry: one shared LRU PlanCache per
// resolved hardware summary. compile() is a pure function and a DiePlan is
// immutable, so every Sys running the same die hardware shares its compiled
// plans instead of recompiling identical plan_cache_keys per rank (design
// doc §8.5: "计划可以缓存"). The simulation is single-threaded (one event
// loop drives all Sys), so the registry needs no lock; if a multi-threaded
// runtime is ever introduced, guard this function and the shared
// DieModel::PlanCache with a mutex.
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
            // Sys::active_Streams already holds every queue id (filled before
            // the SchedulerUnit is constructed); seed each pointer with the
            // corresponding list's begin() instead of a default-constructed
            // (singular) iterator, which must not be copied or assigned.
            this->stream_pointer[base] = sys->active_Streams[base].begin();
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
         AstraNetworkAPI* comm_NI,
         vector<int> physical_dims,
         vector<int> queues_per_dim,
         double injection_scale,
         double comm_scale,
         bool rendezvous_enabled,
         ExecutionDriven::ExecutionMode execution_mode,
         std::shared_ptr<ExecutionDriven::GraphSource> graph_source) {
    // "comm-scale" has no production consumer (its write-only member was
    // retired), so any non-default value is rejected here instead of being
    // silently ignored.
    if (comm_scale != 1.0) {
        sys_panic("unsupported comm-scale value " + std::to_string(comm_scale) +
                  ": only the default 1 is accepted");
    }
    this->execution_mode_ = execution_mode;
    this->graph_source_ = std::move(graph_source);

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
    this->coarse_die_runtime = nullptr;
    this->die_model_profile = "";
    this->die_model_config_path = "";

    this->local_mem_bw = 0;
    this->local_mem_latency = 0;
    this->hbm_bandwidth_contention = true;

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
    this->round_robin_inter_dimension_scheduler = 0;
    this->active_chunks_per_dimension = 1;
    this->priority_counter = 0;
    this->pending_events = 0;
    this->preferred_dataset_splits = 1;

    this->last_scheduled_collective = 0;

    this->first_phase_streams = 0;
    this->total_running_streams = 0;

    this->communication_delay = 10;
    this->local_reduction_delay = 1;

    collective_impl_lookup = new CollectiveImplLookup(id);

    initialize_sys(system_configuration);

    // scheduler
    this->physical_dims = physical_dims;
    // An empty queues_per_dim would skip the >=1 element guard entirely and
    // then index queues_per_dim[0] below (operator[] out of bounds, UB);
    // every real caller passes one entry per physical dimension.
    if (queues_per_dim.empty()) {
        sys_panic("queues_per_dim must carry one entry per physical "
                  "dimension (got an empty vector)");
    }
    // A zero/negative queues-per-dimension entry would make the
    // concurrent_streams division below divide by zero (inf cast to int is
    // UB) and leave that dimension without any queue to schedule into.
    for (int queues_at_dim : queues_per_dim) {
        if (queues_at_dim < 1) {
            sys_panic("queues-per-dimension entries must be >= 1");
        }
    }
    // The loop below indexes physical_dims by queues_per_dim positions;
    // a shorter physical_dims vector read out of bounds here. The only
    // caller passes equal-length vectors, so this is a pure defensive
    // bound -- fail closed with a named message instead.
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

    // Constructor-local scheduler thresholds (2026-09-28: the former data
    // members were never read after this constructor).
    const int concurrent_streams =
        (int)ceil(((double)active_chunks_per_dimension) / queues_per_dim[0]);
    const int active_first_phase = 100000000;
    const int max_running = 100000000;

    scheduler_unit = new SchedulerUnit(this, queues_per_dim, max_running,
                                       active_first_phase, concurrent_streams);

    vLevels = new QueueLevels(queues_per_dim, 0);

    // collective communication
    this->num_streams = 0;

    // GeneralComplexTopology throws on an empty implementation table, more
    // implementations than dimensions, or a non-positive dimension. Route
    // the failure through the unified fail-closed channel instead of letting
    // the exception terminate the process from inside a constructor.
    // Only native implementations exist since the custom collectives / ET
    // feeder path was retired (its keys fail closed at parse time), so the
    // native (BYPASS_ALL_CUSTOM) table is the full table. A collective type
    // with NO implementation at all gets no topology entry (the generate_*
    // lookups and CommunicatorGroup::get_collective_plan all resolve their
    // topology through Sys::get_logical_topology, which fails closed on the
    // missing entry -- naming the type and the absent system key -- before
    // any topology consumer can dereference it).
    try {
        const std::pair<ComType, const char*> coll_types[] = {
            {ComType::All_Reduce, "AllReduce"},
            {ComType::Reduce_Scatter, "ReduceScatter"},
            {ComType::All_Gather, "AllGather"},
            {ComType::All_to_All, "AllToAll"},
        };
        for (const auto& [comm_type, key] : coll_types) {
            std::vector<CollectiveImpl*> impls =
                collective_impl_lookup->get_collective_impl(
                    comm_type, 0, BypassRule::BYPASS_ALL_CUSTOM);
            if (impls.empty()) {
                continue;  // no implementation registered for this type
            }
            // Full-dimension gate (2026-10-01): GeneralComplexTopology
            // tolerates fewer implementations than physical dimensions, but
            // the OfflineGreedy schedule model is built over the FULL
            // physical dim table -- a short impl list desynchronized the
            // load-balancing decisions from the actual topology (chunks
            // "assigned" to dimensions the topology never consumes). Every
            // shipped configuration lists one implementation per dimension;
            // require exactly that here instead of letting the two models
            // silently diverge.
            if (impls.size() != physical_dims.size()) {
                sys_panic(std::string("logical topology for ") + key +
                          " carries " + std::to_string(impls.size()) +
                          " implementation(s) but the system has " +
                          std::to_string(physical_dims.size()) +
                          " physical dimension(s): provide exactly one "
                          "implementation per dimension");
            }
            logical_topologies[key] = new GeneralComplexTopology(
                id, physical_dims, impls);
        }
    } catch (const std::exception& e) {
        sys_panic(std::string("failed to build logical topologies: ") +
                  e.what());
    }

    memBus = new MemBus("NPU", "MA", this, inp_L, inp_o, inp_g, inp_G,
                        model_shared_bus, communication_delay, true);

    // step-1-2 execution-mode factory: online mode injects the dynamic
    // GraphSource at Sys creation (never constructs the ETFeeder, never
    // requires .et files). The offline static path is retired in this
    // repository; Online is the only constructible mode.
    workload = new Workload(this, workload_configuration,
                            comm_group_configuration, execution_mode_,
                            graph_source_);

    if (inter_dimension_scheduling ==
        InterDimensionScheduling::OfflineGreedy) {
        offline_greedy = new OfflineGreedy(this);
    }
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

    // Teardown cancels COMP and endpoint jobs at the current simulation
    // clock. Keep this Sys reachable until Workload has drained them.
    all_sys[id] = nullptr;

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
        exit_sim_loop("Exiting");
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

    // Fail-closed configuration parsing (same channel as every other fatal
    // configuration path): nlohmann throws parse_error for a malformed file
    // and type_error for a field of the wrong JSON type. Neither may escape
    // as an uncaught exception from inside the Sys constructor.
    try {
        initialize_sys_fields(name, inFile);
    } catch (const json::exception& e) {
        sys_panic("system configuration '" + name +
                  "' is not valid JSON or contains a field of the wrong "
                  "type: " +
                  e.what());
    }
    inFile.close();
    return true;
}

void Sys::initialize_sys_fields(const string& name, ifstream& inFile) {
    json j;
    inFile >> j;
    // Die-backend resolution state: the compute-cost-model keys are parsed
    // inside the walk below; the conflict matrix and the die hardware load
    // (initialize_die_backend below) run after the walk so they see the
    // fully parsed roofline-enabled / peak-perf / local-mem-bw state.
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
    }
    if (j.contains("inter-dimension-scheduling")) {
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
        // Negative-domain guard (2026-09-30): LogGP's Processing_Finished
        // delay consumes this straight as a Tick (uint64) delta, so a
        // negative literal would wrap to ~1.8e19 ns and hang the run with
        // no message naming this key -- the same failure family documented
        // at local-mem-latency below.
        const auto& lrd_value = j["local-reduction-delay"];
        if (!lrd_value.is_number_integer() &&
            !lrd_value.is_number_unsigned()) {
            sys_panic("local-reduction-delay must be a non-negative integer "
                      "number of ns in '" +
                      name + "'");
        }
        const int64_t parsed_lrd = lrd_value.get<int64_t>();
        if (parsed_lrd < 0 ||
            parsed_lrd > std::numeric_limits<int>::max()) {
            sys_panic("local-reduction-delay out of domain [0, int-max] in '" +
                      name + "' (got " + lrd_value.dump() + ")");
        }
        local_reduction_delay = static_cast<int>(parsed_lrd);
    }
    if (j.contains("active-chunks-per-dimension")) {
        // Strict type/domain parse (same named-boundary discipline as the
        // neighboring numeric keys): nlohmann's lexical conversion silently
        // truncated a fractional literal (1.5) to 1 with no diagnostic.
        const auto& chunks_value = j["active-chunks-per-dimension"];
        if (!chunks_value.is_number_integer() &&
            !chunks_value.is_number_unsigned()) {
            sys_panic("active-chunks-per-dimension must be an integer in '" +
                      name + "'");
        }
        const int64_t parsed_chunks = chunks_value.get<int64_t>();
        if (parsed_chunks < 1 ||
            parsed_chunks > std::numeric_limits<int>::max()) {
            sys_panic("active-chunks-per-dimension out of domain "
                      "[1, int-max] in '" +
                      name + "' (got " + chunks_value.dump() + ")");
        }
        active_chunks_per_dimension = static_cast<int>(parsed_chunks);
    }
    // A non-positive value would drive concurrent_streams (and therefore the
    // SchedulerUnit's per-queue threshold) to <= 0: no stream would ever be
    // initialized and the run would hang in a misleading protocol dead end.
    if (active_chunks_per_dimension < 1) {
        sys_panic("active-chunks-per-dimension must be >= 1 (got " +
                  std::to_string(active_chunks_per_dimension) + " in '" +
                  name + "')");
    }
    // LogGP integer-domain helper (2026-10-01): MemBus/LogGP declare their
    // L/o/g constructor parameters as Tick (unsigned integer nanoseconds),
    // while this parser holds them as float -- a fractional configured value
    // (e.g. "L": 0.5) used to be silently TRUNCATED to 0 at the memBus
    // construction site with no diagnostic. Reject fractional values here,
    // at the boundary, instead of changing the downstream integer types
    // (which would alter arithmetic on every existing integral input's
    // timing path for no benefit).
    const auto require_integral_ns = [&name](const double value,
                                             const char* key) {
        if (value != std::floor(value)) {
            sys_panic(std::string(key) +
                      " must be a whole number of nanoseconds in '" + name +
                      "' (fractional LogGP latencies are truncated by the "
                      "Tick-typed bus parameters; got " +
                      std::to_string(value) + ")");
        }
    };
    if (j.contains("L")) {
        // Strict parse for the LogGP bus parameters (same family as
        // endpoint-delay below): a non-number literal used to surface only
        // as a generic json type error that did not name the key, and a
        // negative value would wrap when MemBus converts it to a Tick.
        const auto& l_value = j["L"];
        if (!l_value.is_number() || !std::isfinite(l_value.get<double>()) ||
            l_value.get<double>() < 0.0) {
            sys_panic("L must be a finite non-negative number in '" + name +
                      "'");
        }
        require_integral_ns(l_value.get<double>(), "L");
        inp_L = l_value.get<float>();
    }
    if (j.contains("o")) {
        const auto& o_value = j["o"];
        if (!o_value.is_number() || !std::isfinite(o_value.get<double>()) ||
            o_value.get<double>() < 0.0) {
            sys_panic("o must be a finite non-negative number in '" + name +
                      "'");
        }
        require_integral_ns(o_value.get<double>(), "o");
        inp_o = o_value.get<float>();
    }
    if (j.contains("g")) {
        const auto& g_value = j["g"];
        if (!g_value.is_number() || !std::isfinite(g_value.get<double>()) ||
            g_value.get<double>() < 0.0) {
            sys_panic("g must be a finite non-negative number in '" + name +
                      "'");
        }
        require_integral_ns(g_value.get<double>(), "g");
        inp_g = g_value.get<float>();
    }
    if (j.contains("G")) {
        const auto& G_value = j["G"];
        if (!G_value.is_number() || !std::isfinite(G_value.get<double>()) ||
            G_value.get<double>() < 0.0) {
            sys_panic("G must be a finite non-negative number in '" + name +
                      "'");
        }
        inp_G = G_value.get<float>();
    }
    if (j.contains("endpoint-delay")) {
        // injection-scale stretches the endpoint delay uniformly: read the
        // JSON value in the double domain and round once instead of the old
        // double int truncation (JSON 3.5 and scale 2 lost 0.5 ns per
        // message). The key-missing default (10) takes the same scale, so a
        // scaled run no longer mixes two conventions in one process.
        // Negative-domain guard (2026-09-30): MemBus registers the scaled
        // value as a Tick (uint64) delta, so a negative result would wrap
        // to ~1.8e19 ns and hang the run with no message naming this key.
        const auto& delay_value = j["endpoint-delay"];
        if (!delay_value.is_number()) {
            sys_panic("endpoint-delay must be a finite non-negative number "
                      "of ns in '" +
                      name + "'");
        }
        const double scaled =
            delay_value.get<double>() * injection_scale;
        if (!std::isfinite(scaled) || scaled < 0.0 ||
            scaled > static_cast<double>(std::numeric_limits<int>::max())) {
            sys_panic("endpoint-delay out of domain [0, int-max] ns after "
                      "injection scaling in '" +
                      name + "' (scaled value " + std::to_string(scaled) +
                      ")");
        }
        communication_delay = (int)std::llround(scaled);
    } else {
        communication_delay = (int)std::llround(10 * injection_scale);
    }
    if (j.contains("model-shared-bus")) {
        // Boolean/integer strict parsing (same channel as roofline-enabled
        // and hbm-bandwidth-contention below): the former `== 1` comparison
        // silently read any other nonzero integer (e.g. 2, semantically
        // "on") as disabled -- fail-open with no diagnostic.
        const auto& shared_bus = j["model-shared-bus"];
        if (shared_bus.is_boolean()) {
            model_shared_bus = shared_bus.get<bool>();
        } else if (shared_bus.is_number_integer() ||
                   shared_bus.is_number_unsigned()) {
            model_shared_bus = shared_bus.get<int64_t>() != 0;
        } else {
            sys_panic("model-shared-bus must be boolean or integer");
        }
    } else {
        model_shared_bus = false;
    }
    if (j.contains("preferred-dataset-splits")) {
        // Strict type/domain parse: the value feeds the chunk-size division
        // in determine_chunk_size, and a non-positive or fractional literal
        // used to fall through silently to that function's one-chunk
        // fallback with no diagnostic naming this key.
        const auto& splits_value = j["preferred-dataset-splits"];
        if (!splits_value.is_number_integer() &&
            !splits_value.is_number_unsigned()) {
            sys_panic("preferred-dataset-splits must be an integer in '" +
                      name + "'");
        }
        const int64_t parsed_splits = splits_value.get<int64_t>();
        if (parsed_splits < 1 ||
            parsed_splits > std::numeric_limits<int>::max()) {
            sys_panic("preferred-dataset-splits out of domain [1, int-max] "
                      "in '" +
                      name + "' (got " + splits_value.dump() + ")");
        }
        preferred_dataset_splits = static_cast<int>(parsed_splits);
    }
    if (j.contains("peak-perf")) {
        // Domain guard (2026-10-01): peak_perf divides on the compute-timing
        // path, and an explicitly non-positive value constructed a Roofline
        // whose get_perf is <= 0 -- the downstream divisions produced
        // negative/inf runtimes with no diagnostic naming this key. The
        // workload layer keeps its own use-time guard for the key-missing
        // default of 0; an explicitly configured non-positive peak-perf is
        // rejected here, at the boundary.
        const auto& peak_value = j["peak-perf"];
        if (!peak_value.is_number()) {
            sys_panic("peak-perf must be a positive number of TFLOPS in '" +
                      name + "'");
        }
        const double scaled_peak = peak_value.get<double>() * 1e12;
        if (!std::isfinite(scaled_peak) || scaled_peak <= 0.0) {
            sys_panic("peak-perf must be positive (TFLOPS) in '" + name +
                      "' (got " + peak_value.dump() + ")");
        }
        peak_perf = scaled_peak;  // TFLOPS
    }
    if (j.contains("local-mem-bw")) {
        // Domain guard (2026-10-01), symmetric with peak-perf above: the
        // closed-form roofline arm divides by local_mem_bw, so a
        // pathologically small-but-positive value (e.g. 1e-300 GB/s)
        // survives the "<= 0 disables the fluid model" check below yet
        // overflows elapsed_time to inf in the closed-form cast; NaN hides
        // behind every "<= 0" comparison. Reject non-finite and
        // non-positive values at the boundary with this key named.
        const auto& bw_value = j["local-mem-bw"];
        if (!bw_value.is_number()) {
            sys_panic("local-mem-bw must be a positive number of GB/sec "
                      "in '" +
                      name + "'");
        }
        const double scaled_bw = bw_value.get<double>() * 1000000000;
        if (!std::isfinite(scaled_bw) || scaled_bw <= 0.0) {
            sys_panic("local-mem-bw must be positive (GB/sec) in '" + name +
                      "' (got " + bw_value.dump() + ")");
        }
        local_mem_bw = scaled_bw;  // GB/sec -> B/sec
    }
    if (j.contains("local-mem-latency")) {
        // Strict latency parse (same panic discipline as the neighboring
        // fields): nlohmann converts the JSON value lexically, so a
        // negative literal (-100) reinterpreted as uint64_t became
        // ~1.8e19 ns -- the roofline closed-form path computed now+runtime
        // with it, wrapped the event clock, and tripped the EventQueue
        // strict-increase abort (whose message does not name this field);
        // on the fluid HBM path the same value re-scheduled transfer
        // events into the past until the watchdog killed the run. A
        // non-integer or out-of-domain value must fail here, at the
        // boundary, naming the field.
        const auto& latency_value = j["local-mem-latency"];
        if (!latency_value.is_number_integer() &&
            !latency_value.is_number_unsigned()) {
            sys_panic("local-mem-latency must be a non-negative integer "
                      "number of ns in '" +
                      name + "'");
        }
        const int64_t parsed_latency = latency_value.get<int64_t>();
        if (parsed_latency < 0 ||
            parsed_latency > static_cast<int64_t>(1) << 62) {
            sys_panic("local-mem-latency out of domain [0, 2^62] ns (got " +
                      std::to_string(parsed_latency) + " in '" + name + "')");
        }
        local_mem_latency = static_cast<uint64_t>(parsed_latency);
    }
    if (j.contains("roofline-enabled")) {
        // Boolean/integer strict parsing: nlohmann's cross-type `!= 0`
        // would silently treat a JSON `false` as enabled (fail-open).
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
    // Multi-user local-HBM bandwidth contention (hbm-bandwidth-contention,
    // bool/int, default true). Parsed after local-mem-bw so the zero-rate
    // guard below sees the final bandwidth value.
    if (j.contains("hbm-bandwidth-contention")) {
        const auto& contention = j["hbm-bandwidth-contention"];
        if (contention.is_boolean()) {
            hbm_bandwidth_contention = contention.get<bool>();
        } else if (contention.is_number_integer() ||
                   contention.is_number_unsigned()) {
            hbm_bandwidth_contention =
                contention.get<int64_t>() != 0;
        } else {
            sys_panic("hbm-bandwidth-contention must be boolean or integer");
        }
    }
    // coarse_die_v2 derives its HBM rows after this key walk. Preserve
    // the user flag until its descriptor is loaded and validated.
    if (local_mem_bw <= 0 &&
        !(j.contains("die-model-profile") &&
          j["die-model-profile"] == "coarse_die_v2")) {
        // Zero-rate guard: a fluid model over a zero local-mem-bw would
        // never move a byte; fall back to the legacy behavior instead.
        hbm_bandwidth_contention = false;
    }
    // Compute-cost backend keys (contract §3.2). The three keys are parsed
    // here; the conflict matrix and the die hardware load run after the
    // whole key-walk so they can see the fully parsed roofline-enabled /
    // peak-perf / local-mem-bw state.
    this->compute_cost_model = ComputeCostModel::LegacyDefault;
    explicit_compute_cost_model_present = j.contains("compute-cost-model");
    if (explicit_compute_cost_model_present) {
        if (!j["compute-cost-model"].is_string()) {
            Sys::sys_panic(
                "system config key 'compute-cost-model' must be a string");
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
                "system config key 'die-model-profile' must be a string");
        }
        die_model_profile = j["die-model-profile"];
        this->die_model_profile = die_model_profile;
        if (die_model_profile != "mesh_die_v1" &&
            die_model_profile != "tx8_reference_v1" &&
            die_model_profile != "coarse_die_v2") {
            sys_panic("unknown value for die-model-profile in sys input "
                      "file: '" + die_model_profile + "' (legal: "
                      "mesh_die_v1 | tx8_reference_v1 | coarse_die_v2)");
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
    // Compute cost backend resolution (contract §3.2 conflict matrix,
    // fail-closed on every branch -- field parse order must never select a
    // backend silently). This repository removed the replay-only dispatch
    // route (Path-2 removal), so the matrix has no replay-only term; die
    // only conflicts with an enabled roofline here.
    if (explicit_compute_cost_model_present &&
        explicit_compute_cost_model == ComputeCostModel::Die) {
        if (roofline_enabled) {
            sys_panic("system config conflict: compute-cost-model=die is "
                      "incompatible with roofline-enabled=1 (contract 3.2)");
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

    // Workspace admission is an explicit bridge capability. The default
    // `auto` keeps legacy profiles on schema v1 while making coarse_die_v2
    // require the full workspace view/snapshot protocol. Explicit overrides
    // are useful for legacy deployments, but cannot silently disable
    // admission for a coarse backend or enable v2 without one.
    const char* const workspace_mode_env =
        std::getenv("SH_WORKSPACE_ADMISSION");
    const std::string workspace_mode =
        workspace_mode_env == nullptr ? "auto" : workspace_mode_env;
    if (workspace_mode != "auto" && workspace_mode != "required" &&
        workspace_mode != "off") {
        sys_panic("SH_WORKSPACE_ADMISSION must be auto|required|off (got '" +
                  workspace_mode + "')");
    }
    const bool coarse_workspace_backend =
        coarse_die_runtime != nullptr && workspace_admission != nullptr;
    if (workspace_mode == "required" && !coarse_workspace_backend) {
        sys_panic("SH_WORKSPACE_ADMISSION=required needs the coarse_die_v2 "
                  "compute-cost backend");
    }
    if (workspace_mode == "off" && coarse_workspace_backend) {
        sys_panic("SH_WORKSPACE_ADMISSION=off is incompatible with "
                  "coarse_die_v2; refusing to issue Plans without "
                  "workspace admission");
    }

    // Unknown-key gate (3.12, 2026-09-29): this reader only consumed known
    // keys and silently ignored everything else, so a typo'd template key
    // ("endpoint-dely", "roofline-enable") fell back to the built-in
    // default and the feature silently turned off with zero diagnostics --
    // fail-open at the last line of defense. The hardware json side already
    // enforces an exact-key whitelist (_require_exact_keys in the resolver);
    // this is the matching gate for the system side. The retired
    // *-implementation-custom keys are listed as known so they keep their
    // specific reject_retired_custom_key diagnosis inside
    // setup_collective_impl_from_config (called right after this gate)
    // instead of the generic unknown-key message. Keep this set in sync
    // with _KNOWN_SYSTEM_TEMPLATE_KEYS in sh_test_mesh/config_resolver.py
    // (the earliest boundary, same contract).
    static const std::set<std::string> known_system_keys = {
        "scheduling-policy",
        "inter-dimension-scheduling",
        "intra-dimension-scheduling",
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
        "roofline-enabled",
        "hbm-bandwidth-contention",
        "trace-enabled",
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
        "compute-cost-model",
        "die-model-profile",
        "die-model-config",
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

    // coarse_die_v2 has an independent resolved-descriptor schema and
    // shared event runtime. It must be selected explicitly and never enters
    // the old scalar DieModel/roofline machine.
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
        workspace_admission =
            std::make_unique<WorkspaceAdmission>(this->id);
        die_runtime.reset();
        die_model_profile = "coarse_die_v2";
        // Keep the legacy scalar slot only as a compatibility report value;
        // v2 scheduling reads the two typed row rates from Hardware.
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
    // disagrees is a resolver/config regression -- fail closed.
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
    : hardware(std::move(hww)),
      plan_cache(shared_die_plan_cache(hardware)),
      plan_cache_hardware_key(DieModel::plan_cache_hardware_key(hardware)) {}

Tick Sys::boostedTick() {
    if (all_sys.empty()) {
        sys_panic("boostedTick called before any Sys object exists");
    }
    Sys* ts = all_sys[0];
    if (ts == nullptr) {
        bool found = false;
        for (uint64_t i = 1; i < all_sys.size(); i++) {
            if (all_sys[i] != nullptr) {
                ts = all_sys[i];
                found = true;
                break;
            }
        }
        if (!found) {
            sys_panic("boostedTick called with no live Sys object");
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

// Sys itself is never an event target: this override only satisfies the
// Callable interface. System events reach Sys through handleEvent() ->
// call_events() instead, so the body is intentionally empty.
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
            // Fail closed (2026-10-01 tightening): an exception here leaves
            // the stream / DataSet / metric state at an arbitrary midpoint of
            // the callback, and consuming the event as if it had succeeded
            // would let the simulation run on silently corrupt. Paths that
            // must never throw fail with critical + exit at the source; a
            // throw reaching this catch is a bug and stops the run with a
            // diagnosis. (An earlier revision kept an upstream-style
            // swallow-and-continue arm for the static mode behind an
            // execution_mode_ check -- unreachable in this repo, where the
            // Workload constructor fails closed on any non-Online mode; the
            // dead arm is gone so no future mode can silently swallow.)
            LoggerFactory::get_logger("system")
                ->critical("event callback raised std::exception at tick "
                           "{}: {}; failing closed",
                           now, e.what());
            std::exit(EXIT_FAILURE);
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
    // Fail-closed on delta==0 (latent-trap guard): the outer alarm goes
    // through the backend's cancellable schedule, which clamps a delta==0
    // request to +1 ns OUTSIDE invoke context and merges at current_time
    // INSIDE it -- two different instants -- while this Sys-side bucket is
    // keyed at boostedTick()+delta unconditionally. A delta==0 caller
    // therefore got an alarm and an event bucket on two different
    // calendars (orphaned events, silently dropped wakeups) depending on
    // which context the call ran in. The only caller clamps to >= 1
    // already, so the guard costs nothing today and turns the ambiguous
    // request into a named error instead.
    if (delta_cycles == 0) {
        sys_panic("register_event_cancellable requires delta >= 1 ns (a "
                  "delta of 0 is ambiguous across invoke contexts and was "
                  "never a supported request)");
    }
    // Payload-ownership guard (2026-10-01): cancellation bypasses the normal
    // dispatch chain, so the pending CallData is freed ONLY through the
    // cancellation callback. A (null callback, non-null payload) event would
    // leak the payload on every cancel_event; callers with no payload pass
    // nullptr for both.
    if (cancellation_callback == nullptr && callData != nullptr) {
        sys_panic("register_event_cancellable requires a cancellation "
                  "callback when callData is non-null: cancel_event frees "
                  "the pending payload only through that callback");
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
        // Invariant: at most one consumer is armed per payload. The workload
        // arm frees wlhd inside call(), so a second armed consumer would be
        // a use-after-free. Fail closed instead.
        const int armed = (rcehd->workload != nullptr) +
                          (rcehd->owner != nullptr);
        if (armed > 1) {
            LoggerFactory::get_logger("system")
                ->critical("PacketReceived payload armed with {} consumers "
                           "on sys {}",
                           armed, id);
            std::exit(EXIT_FAILURE);
        }
        if (rcehd->workload) {
            rcehd->workload->call(event, rcehd->wlhd);
        }
        if (rcehd->owner) {
            rcehd->owner->consume(rcehd);
        }
        delete rcehd;
    } else if (event == EventType::PacketSent) {
        SendPacketEventHandlerData* sehd = (SendPacketEventHandlerData*)ehd;
        sehd->callable->call(EventType::PacketSent, sehd->wlhd);
        delete sehd;
    } else {
        // Unknown event type: the concrete payload class cannot be recovered
        // from the tag alone, and the handler-data hierarchy has no virtual
        // destructor, so deleting through BasicEventHandlerData* would be
        // undefined behavior. Dropping the event would also strand whichever
        // stream awaited it -- fail closed instead (the leak is moot: the
        // process is exiting).
        Sys::sys_panic("Sys::handleEvent: unknown event type " +
                       std::to_string(static_cast<int>(event)) + " on sys " +
                       std::to_string(id));
    }
}

LogicalTopology* Sys::get_logical_topology(ComType comm_type) {
    const char* table_key;
    const char* config_key;
    if (comm_type == ComType::All_Reduce) {
        table_key = "AllReduce";
        config_key = "all-reduce-implementation";
    } else if (comm_type == ComType::All_to_All) {
        table_key = "AllToAll";
        config_key = "all-to-all-implementation";
    } else if (comm_type == ComType::Reduce_Scatter) {
        table_key = "ReduceScatter";
        config_key = "reduce-scatter-implementation";
    } else if (comm_type == ComType::All_Gather) {
        table_key = "AllGather";
        config_key = "all-gather-implementation";
    } else {
        sys_panic("no known logical topology!");
        return nullptr;
    }
    // F9 (DB2 audit 2026-09-29): the constructor registers a logical
    // topology ONLY for a collective type whose native implementation table
    // is non-empty -- i.e. the system JSON carries a NON-ARRAY-EMPTY
    // '<type>-implementation' key. A bare operator[] lookup on a missing
    // entry would even INSERT a null entry, and generate_collective
    // dereferences the pointer at topology->get_num_of_dimensions() -- a
    // bare segfault with no diagnosis. Fail closed instead, naming the
    // collective type and the missing/empty system-input key. Every logical
    // topology consumer funnels through this lookup: the four world-group
    // generate_* branches and every CommunicatorGroup plan path
    // (CommunicatorGroup::get_collective_plan gates its shaped /
    // whole-cluster / sub-cluster branches through this lookup,
    // 2026-10-01).
    auto entry = logical_topologies.find(table_key);
    if (entry == logical_topologies.end() || entry->second == nullptr) {
        sys_panic(
            std::string("no logical topology registered for collective ") +
            table_key + ": the system input has no non-empty '" +
            config_key +
            "' entry, so this collective cannot run (add the native "
            "implementation key to the system JSON)");
    }
    return entry->second;
}

DataSet* Sys::generate_all_reduce(uint64_t size,
                                  vector<bool> involved_dimensions,
                                  CommunicatorGroup* communicator_group,
                                  int explicit_priority,
                                  uint64_t workload_node_id) {
    if (communicator_group == nullptr) {
        // World-group branch: resolve the logical topology FIRST so a
        // missing native implementation table fails closed through
        // get_logical_topology's named diagnostic, deterministically ahead
        // of the impl lookup's generic numeric-ComType throw (F9).
        LogicalTopology* topology = get_logical_topology(ComType::All_Reduce);
        vector<CollectiveImpl*> implementation_per_dimension;
        implementation_per_dimension = collective_impl_lookup->get_collective_impl(ComType::All_Reduce, workload_node_id);
        return generate_collective(size, topology,
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
        // See generate_all_reduce: topology first, named fail-closed.
        LogicalTopology* topology = get_logical_topology(ComType::All_to_All);
        vector<CollectiveImpl*> implementation_per_dimension;
        implementation_per_dimension = collective_impl_lookup->get_collective_impl(ComType::All_to_All, workload_node_id);
        return generate_collective(size, topology,
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
        // See generate_all_reduce: topology first, named fail-closed.
        LogicalTopology* topology = get_logical_topology(ComType::All_Gather);
        vector<CollectiveImpl*> implementation_per_dimension;
        implementation_per_dimension = collective_impl_lookup->get_collective_impl(ComType::All_Gather, workload_node_id);
        return generate_collective(size, topology,
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
        // See generate_all_reduce: topology first, named fail-closed.
        LogicalTopology* topology =
            get_logical_topology(ComType::Reduce_Scatter);
        vector<CollectiveImpl*> implementation_per_dimension;
        implementation_per_dimension = collective_impl_lookup->get_collective_impl(ComType::Reduce_Scatter, workload_node_id);
        return generate_collective(size, topology,
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
    // The 'preferred-dataset-splits' value from the system JSON input is
    // intertwined deeply in this function so that we cannot remove it for
    // now. Therefore, we have to keep that value in the JSON input.
    // TODO: Refactor and remove.
    // A zero-size collective cannot be split into streams: ceil(0/chunk) = 0
    // streams would leave the DataSet unfinished forever, and the non
    // All_Gather path in determine_chunk_size would inflate size back up to
    // splits * total_nodes and time the empty collective as real traffic.
    if (size == 0) {
        sys_panic("collective data size must be positive");
    }
    // N14 parser contract: an online graph may omit coll.involved_dim, which
    // arrives here as an empty vector. The legacy GraphSource adapter treated
    // an absent attribute as "all dimensions involved", so pad the missing
    // suffix with true instead of indexing dimensions_involved out of bounds
    // in the per-dimension loops below.
    const int num_of_dimensions = topology->get_num_of_dimensions();
    if ((int)dimensions_involved.size() < num_of_dimensions) {
        dimensions_involved.resize(num_of_dimensions, true);
    }
    uint64_t chunk_size = determine_chunk_size(size, collective_type);
    if (chunk_size == 0) {
        sys_panic("collective chunk size must be positive");
    }
    uint64_t recommended_chunk_size = chunk_size;
    int streams = ceil(((double)size) / chunk_size);
    uint64_t remain_size;
    DataSet* dataset = new DataSet(streams);
    int pri = get_priority(explicit_priority);
    int count = 0;
    if (id == 0 && inter_dimension_scheduling ==
                       InterDimensionScheduling::OfflineGreedy) {
        if (last_scheduled_collective != Sys::boostedTick()) {
            offline_greedy->reset_loads();
            last_scheduled_collective = Sys::boostedTick();
        }
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
                collective_type);
            chunk_size = prev_size - size;
        }

        // P2 (2026-09-28): the tail stream of a splits-not-divisible
        // collective must carry its ACTUAL remaining bytes. The former
        // unconditional remain_size = chunk_size padded the tail chunk to a
        // full chunk_size, systematically overstating the collective's
        // traffic (All_Gather shares the same assignment). The stream COUNT
        // (ceil) is unchanged.
        if (collective_type == ComType::All_to_All ||
            inter_dimension_scheduling !=
                InterDimensionScheduling::OfflineGreedy) {
            if (chunk_size > size) {
                remain_size = size;  // tail block: actual remaining bytes
                size = 0;
            } else {
                remain_size = chunk_size;
                size -= chunk_size;
            }
        } else {
            // OfflineGreedy (non-All_to_All): chunk_size was rewritten above
            // to the exact bytes this iteration consumed, so it already is
            // the tail-accurate remainder.
            remain_size = chunk_size;
        }
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
            // Collective stream ids double as COLLECTIVE tags: the world
            // group owns [0, CommunicatorGroup::STREAM_WINDOW) and the
            // group with logical id N owns [N * STREAM_WINDOW,
            // (N + 1) * STREAM_WINDOW). The window base is DERIVED FROM THE
            // LOGICAL GROUP ID (CommunicatorGroup::set_id ->
            // lease_stream_window), so every rank's copy of the same logical
            // group starts at the same base and -- same graph, same splits,
            // guaranteed by the online commit boundary -- advances in
            // lockstep: cross-rank tag consistency is guaranteed THERE (the
            // lease also fails closed on an id whose window would fold in
            // the COLLECTIVE remap and on a second, differently shaped live
            // group claiming the same id). This guard therefore only catches
            // a single rank exhausting its own window: crossing a boundary
            // would alias another group's (src, dst, tag) triples -- a
            // cross-pairing or a permanent mutual wait -- so fail closed.
            // (num_streams on Sys is only ever read as the world-group
            // counter, so it advances only on the world branch.)
            int stream_id;
            if (communicator_group != nullptr) {
                // Exhaustion check BEFORE the allocation (2026-10-01): the
                // former post-increment comparison panicked after handing
                // out the window's LAST legal id, so the final stream id of
                // every window was permanently unusable (real capacity
                // STREAM_WINDOW - 1) and the "exclusive upper bound" contract
                // lied. Check first, then hand out the id.
                if (communicator_group->num_streams >=
                    communicator_group->stream_tag_ceiling()) {
                    sys_panic("communicator group " +
                              std::to_string(communicator_group->get_id()) +
                              " exhausted its collective stream id window");
                }
                stream_id = communicator_group->num_streams++;
                // Cheap cross-module tie: the lease validates against its
                // own hard-coded copy of the fold point (500000000); this
                // one-compare assert pins the group window to THIS
                // translation unit's COLLECTIVE tag window, so a constant
                // drift on either side fails closed at the first group
                // stream instead of folding silently in the front-end remap.
                if (communicator_group->stream_tag_ceiling() >
                    (int)Sys::FrontEndSendRecvType::COLLECTIVE) {
                    sys_panic("communicator group " +
                              std::to_string(communicator_group->get_id()) +
                              " stream window overflows the COLLECTIVE tag "
                              "window (CommunicatorGroup lease constant and "
                              "Sys::FrontEndSendRecvType::COLLECTIVE "
                              "disagree)");
                }
            } else {
                // Same check-before-allocation for the world-group window.
                if (num_streams >= CommunicatorGroup::STREAM_WINDOW) {
                    sys_panic("world-group collective stream id space "
                              "exhausted (> " +
                              std::to_string(
                                  CommunicatorGroup::STREAM_WINDOW) +
                              " streams)");
                }
                stream_id = num_streams++;
            }
            StreamBaseline* newStream =
                new StreamBaseline(this, dataset, stream_id, vect, pri);
            newStream->current_queue_id = -1;
            insert_into_ready_list(newStream);
        } else {
            // Degenerate collective (every topology dimension is single-node
            // or not involved): zero streams are created, so the DataSet can
            // never reach finished_streams == total_streams through stream
            // completions and the workload node would hang forever. Complete
            // it through the event loop instead (same pattern as the
            // workload's BROADCAST path): one self event acts as the only
            // "stream", so when it pops, notify_stream_finished fires the
            // notifier that the workload layer registers right after this
            // generate_* call returns.
            dataset->active = false;
            dataset->total_streams = 1;
            register_event(dataset, EventType::General, nullptr, 1);
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
    if (collective_impl->type == CollectiveImplType::Ring ||
        collective_impl->type == CollectiveImplType::OneRing) {
        CollectivePhase vn(queue_id,
                           new Ring(collective_type, id,
                                    (RingTopology*)topology, data_size,
                                    direction));
        return vn;
    } else if (collective_impl->type == CollectiveImplType::Direct ||
               collective_impl->type == CollectiveImplType::OneDirect) {
        CollectivePhase vn(queue_id,
                           new AllToAll(collective_type,
                                        ((DirectCollectiveImpl*)collective_impl)
                                            ->direct_collective_window,
                                        id, (RingTopology*)topology, data_size,
                                        direction));
        return vn;
    } else if (collective_impl->type == CollectiveImplType::DoubleBinaryTree) {
        // A misconfigured pairing (DBT implementation over a non-tree
        // topology) must fail loudly instead of reading garbage through a
        // C-style downcast.
        BinaryTree* btree = dynamic_cast<BinaryTree*>(topology);
        if (btree == nullptr) {
            sys_panic("DoubleBinaryTree requires a BinaryTree logical "
                      "topology");
        }
        // F-RV8-1 (2026-09-29): DoubleBinaryTreeAllReduce hardcodes
        // All_Reduce semantics (the algorithm class has no ComType
        // parameter), so a doubleBinaryTree entry under
        // all-gather/reduce-scatter-implementation would silently simulate
        // the wrong collective and still "complete" -- the same family as
        // the plain-Ring-drives-All_to_All guard in Ring::run. Fail closed
        // at phase generation instead.
        if (collective_type != ComType::All_Reduce) {
            const char* com_type_name =
                collective_type == ComType::All_Gather       ? "All_Gather"
                : collective_type == ComType::Reduce_Scatter ? "Reduce_Scatter"
                : collective_type == ComType::All_to_All     ? "All_to_All"
                                                             : "None";
            sys_panic("DoubleBinaryTree implements All_Reduce only; it is "
                      "configured for " +
                      std::string(com_type_name) +
                      " -- use ring/direct for that collective");
        }
        CollectivePhase vn(queue_id,
                           new DoubleBinaryTreeAllReduce(id, btree, data_size));
        return vn;
    } else if (collective_impl->type == CollectiveImplType::HalvingDoubling ||
               collective_impl->type ==
                   CollectiveImplType::OneHalvingDoubling) {
        CollectivePhase vn(queue_id,
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
        // Zero/negative splits must not reach the division below; fall back
        // to one chunk covering the whole size. Defense in depth: the
        // parser already rejects an explicitly non-positive
        // preferred-dataset-splits, and the default when the key is absent
        // is 1.
        return size;
    }
    uint64_t chunk_size = size / preferred_dataset_splits;
    if (type == ComType::All_Gather && chunk_size == 0) {
        // An All_Gather payload smaller than 'preferred-dataset-splits'
        // divides into zero-size chunks; ceil(size/0) would then overflow
        // the stream count. Fall back to one chunk covering the whole size.
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

// Test-only entry point (see the comment on the declaration in Sys.hh);
// production scheduling goes through SchedulerUnit::notify_stream_added*.
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
        // Null guard (2026-10-01): ~Sys parks nullptr in its all_sys slot,
        // so once any rank is destroyed the shared-slot iteration must skip
        // it instead of dereferencing (test-only reachability today: the
        // sole caller is system_history_lifecycle_test, but the loop is
        // shared-state code and the guard is free).
        if (sys == nullptr) {
            continue;
        }
        if (sys->ready_list.size() == 0 ||
            sys->ready_list.front()->stream_id != top) {
            return;
        }
        if (sys->ready_list.size() < min) {
            min = sys->ready_list.size();
        }
    }
    for (auto& sys : all_sys) {
        // Same null guard as the scan loop above: ~Sys parks nullptr in
        // its all_sys slot (B1 follow-up, 2026-10-01 -- the guard was
        // initially added to the first loop only).
        if (sys == nullptr) {
            continue;
        }
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
    if (stream->my_current_phase.algorithm != nullptr) {
        delete stream->my_current_phase.algorithm;
    }
    if (stream->phases_to_go.size() == 0) {
        // The take_bus_stats_average() call here used to average the
        // stream's (write-only, zero-reader) bus stats before notifying the
        // dataset; the whole bus-stat accounting was removed on 2026-10-01
        // (face parity), so the stream finishes with no stat side effect.
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
        // level with zero queues; the Sys constructor already rejects
        // queues-per-dim entries < 1). Flipping the sign would silently
        // occupy an unrelated queue of another dimension -- the
        // cross-dimension queue-sharing race documented in the issue #137
        // comment in generate_collective.
        sys_panic(
            "negative queue id (" +
            std::to_string(stream->phases_to_go.front().queue_id) +
            ") in a collective phase: the phase was assigned a queue that "
            "does not exist");
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
    // The remap below assumes a tag inside the NATIVE/COLLECTIVE window
    // [0, COLLECTIVE). A tag at or beyond the window would wrap around the
    // modulo and alias another stream's tag (cross-pairing or a permanent
    // mutual wait) -- fail closed instead of folding silently.
    // Cross-rank consistency is NOT this guard's job: collective tags are
    // stream ids drawn from the CommunicatorGroup id-derived window (see
    // CommunicatorGroup::set_id / lease_stream_window), identical on every
    // rank's copy of the same logical group, so the same collective step
    // emits the same tag on every rank by construction. This guard only
    // stops a locally malformed tag from folding into another stream's
    // tag space.
    if (tag < 0 || tag >= (int)Sys::FrontEndSendRecvType::COLLECTIVE) {
        sys_panic("front-end send tag " + std::to_string(tag) +
                  " is outside the [0, " +
                  std::to_string((int)Sys::FrontEndSendRecvType::COLLECTIVE) +
                  ") window");
    }
    if (send_type == Sys::FrontEndSendRecvType::NATIVE) {
        tag = tag % (Sys::FrontEndSendRecvType::COLLECTIVE -
                     Sys::FrontEndSendRecvType::NATIVE) +
              Sys::FrontEndSendRecvType::NATIVE;
    } else if (send_type == Sys::FrontEndSendRecvType::COLLECTIVE) {
        tag = tag % (Sys::FrontEndSendRecvType::RENDEZVOUS -
                     Sys::FrontEndSendRecvType::COLLECTIVE) +
              Sys::FrontEndSendRecvType::COLLECTIVE;
    } else {
        sys_panic("a RENDEZVOUS send type must never be issued through "
                  "front_end_sim_send (the rendezvous handshake is routed "
                  "internally by Sys)");
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
    // Same window guard as front_end_sim_send: never let a tag wrap around
    // the modulo remap into another stream's tag space. As there, cross-rank
    // tag consistency comes from the CommunicatorGroup id-derived stream
    // window (see CommunicatorGroup::set_id / lease_stream_window), not
    // from this local guard.
    if (tag < 0 || tag >= (int)Sys::FrontEndSendRecvType::COLLECTIVE) {
        sys_panic("front-end recv tag " + std::to_string(tag) +
                  " is outside the [0, " +
                  std::to_string((int)Sys::FrontEndSendRecvType::COLLECTIVE) +
                  ") window");
    }
    if (recv_type == Sys::FrontEndSendRecvType::NATIVE) {
        tag = tag % (Sys::FrontEndSendRecvType::COLLECTIVE -
                     Sys::FrontEndSendRecvType::NATIVE) +
              Sys::FrontEndSendRecvType::NATIVE;
    } else if (recv_type == Sys::FrontEndSendRecvType::COLLECTIVE) {
        tag = tag % (Sys::FrontEndSendRecvType::RENDEZVOUS -
                     Sys::FrontEndSendRecvType::COLLECTIVE) +
              Sys::FrontEndSendRecvType::COLLECTIVE;
    } else {
        sys_panic("a RENDEZVOUS recv type must never be issued through "
                  "front_end_sim_recv (the rendezvous handshake is routed "
                  "internally by Sys)");
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
        sys_panic("tag is bigger than FrontEndSendRecvType::RENDEZVOUS, "
                  "which means it might be mistakenly used as a rendezvous "
                  "tag.");
    }
    RendezvousSendData* rsd =
        new RendezvousSendData(id, this, buffer, count, type, dst, tag,
                               *request, msg_handler, fun_arg);
    sim_request newReq = *request;
    uint64_t rendevouz_size = 8192;
    // The rendezvous handshake reverses the direction of the original
    // transfer. Build the swapped identity from the validated function
    // parameters: the caller-supplied request does not reliably carry
    // srcRank/dstRank (collective recv requests leave them unset).
    newReq.dstRank = (uint32_t)this->id;
    newReq.srcRank = (uint32_t)dst;
    newReq.reqCount = rendevouz_size;
    int newTag = tag + Sys::FrontEndSendRecvType::RENDEZVOUS;
    newReq.tag = newTag;
    sim_recv(delay, buffer, rendevouz_size, type, dst, newTag, &newReq,
             &Sys::handleEvent, rsd);
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
        sys_panic("tag is bigger than FrontEndSendRecvType::RENDEZVOUS, "
                  "which means it might be mistakenly used as a rendezvous "
                  "tag.");
    }
    RendezvousRecvData* rrd =
        new RendezvousRecvData(id, this, buffer, count, type, src, tag,
                               *request, msg_handler, fun_arg);
    sim_request newReq = *request;
    uint64_t rendevouz_size = 8192;
    // See rendezvous_sim_send: derive the reversed identity from the
    // function parameters, never from the possibly-uninitialized request
    // fields.
    newReq.dstRank = (uint32_t)src;
    newReq.srcRank = (uint32_t)this->id;
    newReq.reqCount = rendevouz_size;
    int newTag = tag + Sys::FrontEndSendRecvType::RENDEZVOUS;
    newReq.tag = newTag;
    sim_send(delay, buffer, rendevouz_size, type, src, newTag, &newReq,
             &Sys::handleEvent, rrd);
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
    // Every call site (frontend p2p and the rendezvous handshake) passes
    // delay == 0; the former delay>0 deferred branch (SimSendCaller event)
    // was unreachable and was removed. Keep a loud guard so a future
    // non-zero delay cannot silently degrade into an immediate send.
    if (delay != 0) {
        sys_panic("Sys::sim_send with a non-zero delay is not supported");
    }
    comm_NI->sim_send(buffer, count, type, dst, tag, request, msg_handler,
                      fun_arg);
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
    // See sim_send: delay>0 was an unreachable dead branch.
    if (delay != 0) {
        sys_panic("Sys::sim_recv with a non-zero delay is not supported");
    }
    comm_NI->sim_recv(buffer, count, type, src, tag, request, msg_handler,
                      fun_arg);
    return 1;
}
}  // namespace AstraSim
