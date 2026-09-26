/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/Sys.hh"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <utility>
#include <iostream>

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/BaseStream.hh"
#include "astra-sim/system/CollectivePlan.hh"
#include "astra-sim/system/DataSet.hh"
#include "astra-sim/system/MemBus.hh"
#include "astra-sim/system/QueueLevels.hh"
#include "astra-sim/system/RendezvousRecvData.hh"
#include "astra-sim/system/RendezvousSendData.hh"
#include "astra-sim/system/SendPacketEventHandlerData.hh"
#include "astra-sim/system/SimRecvCaller.hh"
#include "astra-sim/system/SimSendCaller.hh"
#include "astra-sim/system/StreamBaseline.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include "astra-sim/system/astraccl/CollectiveImpl.hh"
#include "astra-sim/system/astraccl/custom_collectives/CustomAlgorithm.hh"
#include "astra-sim/system/astraccl/native_collectives/collective_algorithm/AllToAll.hh"
#include "astra-sim/system/astraccl/native_collectives/collective_algorithm/DoubleBinaryTreeAllReduce.hh"
#include "astra-sim/system/astraccl/native_collectives/collective_algorithm/HalvingDoubling.hh"
#include "astra-sim/system/astraccl/native_collectives/collective_algorithm/Ring.hh"
#include "astra-sim/system/astraccl/native_collectives/logical_topology/BasicLogicalTopology.hh"
#include "astra-sim/system/astraccl/native_collectives/logical_topology/GeneralComplexTopology.hh"
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

// Fail-closed boolean config-flag parsing, shared by the roofline-enabled /
// trace-enabled / replay-only / track-local-mem keys (same discipline as the
// hbm-kv-restore-bandwidth-sharing / hbm-bandwidth-contention keys below).
// The legacy `j[key] != 0` idiom compared across JSON types, so a JSON
// `false` (a different type from any number) read as "enabled" -- the
// opposite of the written intent.
bool parse_bool_flag(const json& value, const string& key) {
    if (value.is_boolean()) {
        return value.get<bool>();
    }
    if (value.is_number_integer() || value.is_number_unsigned()) {
        return value.get<int64_t>() != 0;
    }
    Sys::sys_panic(key + " must be boolean or integer");
    return false;  // unreachable: sys_panic exits
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
         AstraRemoteMemoryAPI* remote_mem,
         AstraNetworkAPI* comm_NI,
         vector<int> physical_dims,
         vector<int> queues_per_dim,
         double injection_scale,
         double /*comm_scale*/,  // member removed (zero readers); the CLI
                                 // argument is kept for call-site compat
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

    this->workload = nullptr;

    this->roofline_enabled = false;
    this->peak_perf = 0;
    this->roofline = nullptr;

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

    this->comm_NI = comm_NI;
    this->rendezvous_enabled = rendezvous_enabled;

    this->scheduler_unit = nullptr;
    this->vLevels = nullptr;
    this->scheduling_policy = SchedulingPolicy::FIFO;
    this->active_chunks_per_dimension = 1;
    this->priority_counter = 0;
    this->pending_events = 0;
    this->preferred_dataset_splits = 1;
    this->collectiveOptimization = CollectiveOptimization::Baseline;

    this->first_phase_streams = 0;
    this->total_running_streams = 0;

    this->communication_delay = 10;
    this->local_reduction_delay = 1;

    collective_impl_lookup = new CollectiveImplLookup(id);

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

    if (queues_per_dim.empty() || queues_per_dim[0] <= 0) {
        // queues_per_dim comes from the frontend dimension list expanded by
        // num-queues-per-dim; the frontend only rejects a malformed list much
        // later (first fluid route), so fail closed here before the division
        // below divides by zero or indexes an empty vector.
        sys_panic("invalid queues-per-dim: at least one positive queue count "
                  "is required to derive concurrent_streams");
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

void Sys::initialize_sys(string name) {
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
    inFile >> j;
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
        local_reduction_delay = j["local-reduction-delay"];
    }
    if (j.contains("active-chunks-per-dimension")) {
        active_chunks_per_dimension = j["active-chunks-per-dimension"];
    }
    if (j.contains("L")) {
        inp_L = j["L"];
    }
    if (j.contains("o")) {
        inp_o = j["o"];
    }
    if (j.contains("g")) {
        inp_g = j["g"];
    }
    if (j.contains("G")) {
        inp_G = j["G"];
    }
    if (j.contains("endpoint-delay")) {
        communication_delay = j["endpoint-delay"];
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
        local_mem_latency = j["local-mem-latency"];  // ns
    }
    if (j.contains("remote-mem-bw")) {
        remote_mem_bw = j["remote-mem-bw"];
        remote_mem_bw = remote_mem_bw * 1000000000;  // GB/sec
    }
    if (j.contains("remote-mem-latency")) {
        remote_mem_latency = j["remote-mem-latency"];  // ns
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
    // N-way HBM contention master switch (default true). Parsed after
    // local-mem-bw so the <=0 auto-disable applies to the final value.
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
    if (local_mem_bw <= 0) {
        // No local HBM bandwidth to share: the N-way model cannot run.
        // Same graceful degradation for the KV-restore sharing flag (no
        // bandwidth to share over; configs with restore nodes are still
        // rejected fail-closed on the Workload side).
        hbm_bandwidth_contention = false;
        hbm_kv_restore_bandwidth_sharing = false;
    }
    if (j.contains("roofline-enabled")) {
        if (parse_bool_flag(j["roofline-enabled"], "roofline-enabled")) {
            roofline_enabled = true;
            roofline = new Roofline(local_mem_bw, peak_perf);
        }
    }
    this->trace_enabled = false;
    if (j.contains("trace-enabled")) {
        this->trace_enabled = parse_bool_flag(j["trace-enabled"],
                                              "trace-enabled");
    }
    this->replay_only = false;
    if (j.contains("replay-only")) {
        this->replay_only = parse_bool_flag(j["replay-only"], "replay-only");
    }
    this->track_local_mem = false;
    if (j.contains("track-local-mem")) {
        this->track_local_mem = parse_bool_flag(j["track-local-mem"],
                                                "track-local-mem");
    }

    this->local_mem_trace_filename = "local_mem_trace";
    if (j.contains("local-mem-trace-filename")) {
        this->local_mem_trace_filename = j["local-mem-trace-filename"];
    }

    collective_impl_lookup->setup_collective_impl_from_config(j);

    inFile.close();
}

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
            if (execution_mode_ == ExecutionDriven::ExecutionMode::Online) {
                // Online mode fails closed (README sec.6 "missing inputs fail
                // closed" discipline): an exception here leaves the stream /
                // DataSet / metric state at an arbitrary midpoint of the
                // callback, and consuming the event as if it had succeeded
                // would let the simulation run on silently corrupt. Paths
                // that must never throw fail with critical + exit at the
                // source; a throw reaching this catch is a bug and stops the
                // run with a diagnosis.
                logger->critical("online event callback raised std::exception "
                                 "at tick {}: {}; failing closed",
                                 now, e.what());
                std::exit(EXIT_FAILURE);
            }
            // Static/legacy compatibility path keeps the upstream
            // swallow-and-continue behavior.
            logger->critical("event callback raised std::exception at tick "
                             "{}: {} (static path continues)",
                             now, e.what());
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
    // TODO(jinsun): For custom collective, we do not need the chunk_size here (since the chunk size is already determined)
    // Therefore, we also do not need the 'preferred-dataset-splits' value from the system JSON input. 
    // However, this variable is intertwined deeply in this function so that we cannot remove it for now.
    // Therefore, we have to keep that value in the JSON input. TODO: Refactor and remove.
    if (size == 0) {
        // A zero-byte collective can never complete: its DataSet would wait
        // for streams that are never created, and the chunking below would
        // divide by zero.
        sys_panic("collective data size must be positive");
    }
    uint64_t chunk_size = determine_chunk_size(size, collective_type);
    int streams = ceil(((double)size) / chunk_size);
    uint64_t remain_size;
    DataSet* dataset = new DataSet(streams);
    int pri = get_priority(explicit_priority);
    int count = 0;

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
            // Below three are default values.
            RingTopology::Direction::Clockwise,
            InjectionPolicy::Normal,
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
        return dataset;
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
            // In this branch, a collective
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
            dataset->active = false;
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
        CollectivePhase vn(queue_id,
                           new Ring(collective_type, id,
                                    (RingTopology*)topology, data_size,
                                    direction, injection_policy));
        return vn;
    } else if (collective_impl->type == CollectiveImplType::Direct ||
               collective_impl->type == CollectiveImplType::OneDirect) {
        CollectivePhase vn(queue_id,
                           new AllToAll(collective_type,
                                        ((DirectCollectiveImpl*)collective_impl)
                                            ->direct_collective_window,
                                        id, (RingTopology*)topology, data_size,
                                        direction, InjectionPolicy::Normal));
        return vn;
    } else if (collective_impl->type == CollectiveImplType::DoubleBinaryTree) {
        CollectivePhase vn(queue_id,
                           new DoubleBinaryTreeAllReduce(
                               id, (BinaryTree*)topology, data_size));
        return vn;
    } else if (collective_impl->type == CollectiveImplType::HalvingDoubling ||
               collective_impl->type ==
                   CollectiveImplType::OneHalvingDoubling) {
        CollectivePhase vn(queue_id,
                           new HalvingDoubling(collective_type, id,
                                               (RingTopology*)topology,
                                               data_size));
        return vn;
    } else if (collective_impl->type == CollectiveImplType::CustomCollectiveImpl) {
        string filename = ((CustomCollectiveImpl*)collective_impl)->filename;
        CollectivePhase vn(0, new CustomAlgorithm(filename, id, queue_id, comm_group));
        return vn;
    } else {
        LoggerFactory::get_logger("system")->critical(
            "Error: No known collective implementation for collective phase");
        exit(1);
    }
}

uint64_t Sys::determine_chunk_size(uint64_t& size, ComType type) {
    if (preferred_dataset_splits <= 0) {
        // Zero/negative splits (e.g. the config key is missing or malformed)
        // must not reach the division below; fall back to one chunk covering
        // the whole size.
        return size;
    }
    uint64_t chunk_size = size / preferred_dataset_splits;
    // We want the collective size to have minimum size, otherwise, there is a
    // possibility of size overflow due to further dividing it to more
    // fine-grained messages
    if (type != ComType::All_Gather && this->total_nodes > chunk_size) {
        chunk_size = this->total_nodes;
        size = preferred_dataset_splits * chunk_size;
    }
    if (chunk_size == 0) {
        // All_Gather skips the total_nodes clamp above, so a message smaller
        // than preferred_dataset_splits yields a zero chunk. The caller
        // divides by the chunk (UB via infinity) and subtracts it in a loop
        // that would then never terminate. One chunk per message is the
        // smallest usable decomposition; size > 0 is guaranteed upstream.
        chunk_size = 1;
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
    if (stream->steps_finished != 0 && stream->net_message_counter != 0) {
        stream->net_message_latency.back() /= stream->net_message_counter;
    }
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
            scheduler_unit->notify_stream_removed(previous_vnet);
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

    CollectivePhase vi = stream->phases_to_go.front();
    stream->my_current_phase = vi;
    stream->phases_to_go.pop_front();
    stream->initialized = false;
    stream->last_phase_change = Sys::boostedTick();

    stream->net_message_latency.push_back(0);
    stream->net_message_counter = 0;

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
    uint64_t rendevouz_size = 8192;
    newReq.dstRank = request->srcRank;
    newReq.srcRank = request->dstRank;
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
