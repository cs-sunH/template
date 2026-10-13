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
#include "astra-sim/system/scheduling/OfflineGreedy.hh"
#include "astra-sim/system/astraccl/native_collectives/logical_topology/BasicLogicalTopology.hh"
#include "astra-sim/system/astraccl/native_collectives/logical_topology/GeneralComplexTopology.hh"
#include "DieModel.hh"
#include "astra-sim/workload/CoarseDieRuntime.hh"
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

// Process-level die plan-cache registry (stage E2 as delivered in the face
// reference): one shared LRU PlanCache per resolved hardware summary.
// compile() is a pure function and a DiePlan is immutable, so every Sys
// running the same die hardware shares its compiled plans instead of
// recompiling identical plan_cache_keys per rank (design doc §8.5: "计划可
// 以缓存"). The simulation is single-threaded (one event loop drives all
// Sys), so the registry needs no lock; if a multi-threaded runtime is ever
// introduced, guard this function and the shared DieModel::PlanCache with
// a mutex.
shared_ptr<DieModel::PlanCache> shared_die_plan_cache(
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

    int base = 0;
    for (auto q : queues) {
        for (int i = 0; i < q; i++) {
            this->running_streams[base] = 0;
            list<BaseStream*>::iterator it;
            this->stream_pointer[base] = it;
            base++;
        }
    }
}

void Sys::SchedulerUnit::notify_stream_added(int vnet) {
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

    this->remote_mem = remote_mem;
    // Every construction site passes a real remote-memory backend; a null
    // here is a wiring bug and must die with a diagnosis instead of a
    // SIGSEGV one line later.
    if (this->remote_mem == nullptr) {
        sys_panic("Sys constructed without a remote-memory backend");
    }
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
    this->local_reduction_delay = 0;

    this->comm_NI = comm_NI;
    this->rendezvous_enabled = rendezvous_enabled;

    this->scheduler_unit = nullptr;
    this->vLevels = nullptr;
    this->offline_greedy = nullptr;
    this->inter_dimension_scheduling = InterDimensionScheduling::Ascending;
    this->scheduling_policy = SchedulingPolicy::FIFO;
    this->round_robin_inter_dimension_scheduler = 0;
    this->active_chunks_per_dimension = 1;
    this->priority_counter = 0;
    this->pending_events = 0;
    this->preferred_dataset_splits = 1;

    this->collectiveOptimization = CollectiveOptimization::Baseline;
    this->last_scheduled_collective = 0;

    this->first_phase_streams = 0;
    this->total_running_streams = 0;

    this->communication_delay = 10;
    this->local_reduction_delay = 1;

    collective_impl_lookup = new CollectiveImplLookup();

    initialize_sys(system_configuration);

    // scheduler
    this->physical_dims = physical_dims;
    this->queues_per_dim = queues_per_dim;
    // A dimension-count mismatch would make the loop below index
    // physical_dims out of range (UB) and silently drop whole dimensions
    // from the scheduler. Fail closed at initialization instead.
    if (physical_dims.size() != queues_per_dim.size()) {
        sys_panic("physical-dims size (" +
                  std::to_string(physical_dims.size()) +
                  ") must match queues-per-dim size (" +
                  std::to_string(queues_per_dim.size()) + ")");
    }
    // A zero/negative queues-per-dimension entry would make the
    // concurrent_streams division below divide by zero (inf cast to int is
    // UB) and leave that dimension without any queue to schedule into.
    // (The empty-vector case cannot reach production: the online entry
    // pushes one entry per topology dimension and the network parser
    // fail-closes on dims < 1 -- guarded here for constructor completeness.)
    if (queues_per_dim.empty()) {
        sys_panic("queues-per-dimension must be non-empty");
    }
    for (int queues_at_dim : queues_per_dim) {
        if (queues_at_dim < 1) {
            sys_panic("queues-per-dimension entries must be >= 1");
        }
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

    // Scheduler thresholds: computed here and handed to the SchedulerUnit,
    // which owns the only live copies (the former Sys-side members had no
    // reader outside the constructor).
    const int concurrent_streams =
        (int)ceil(((double)active_chunks_per_dimension) / queues_per_dim[0]);
    const int active_first_phase = 100000000;
    const int max_running = 100000000;

    scheduler_unit = new SchedulerUnit(this, queues_per_dim, max_running,
                                       active_first_phase, concurrent_streams);

    vLevels = new QueueLevels(queues_per_dim, 0);

    // collective communication
    this->num_streams = 0;

    // GeneralComplexTopology throws when given more implementations than
    // topology dimensions, or a non-positive dimension size (an empty
    // implementation table is tolerated above the lookup). Route the
    // failure through the unified fail-closed channel instead of letting
    // the exception terminate the process from inside a constructor.
    try {
        // 每维恰一实现门（2026-10-01 对齐）：GeneralComplexTopology 本身
        // 容忍实现数少于物理维数，但 OfflineGreedy 的负载模型按完整物
        // 理维表构建——短实现清单会使负载均衡决策与实际拓扑失同步（块
        // 被"分配"到拓扑从不消费的维度）。全部交付配置每维恰列一个实
        // 现；在此要求恰一，而不是让两个模型静默分叉。
        const std::pair<const char*, ComType> impl_tables[] = {
            {"AllReduce", ComType::All_Reduce},
            {"ReduceScatter", ComType::Reduce_Scatter},
            {"AllGather", ComType::All_Gather},
            {"AllToAll", ComType::All_to_All},
        };
        for (const auto& [key, com_type] : impl_tables) {
            const auto impls = collective_impl_lookup->get_collective_impl(
                com_type, 0, BypassRule::BYPASS_ALL_CUSTOM);
            if (!impls.empty() && impls.size() != physical_dims.size()) {
                sys_panic(std::string("logical topology for ") + key +
                          " carries " + std::to_string(impls.size()) +
                          " implementation(s) but the system has " +
                          std::to_string(physical_dims.size()) +
                          " physical dimension(s): provide exactly one "
                          "implementation per dimension");
            }
            logical_topologies[key] =
                new GeneralComplexTopology(id, physical_dims, impls);
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

    json j;
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
    // Key-presence probes are taken AFTER the document parse below (this
    // repository parses j inside the try block; probing an empty document
    // would silently misfire every branch of the backend resolution).
    bool peak_perf_key_present = false;
    bool local_mem_bw_key_present = false;
    bool hbm_contention_key_present = false;
    // Every JSON read / type conversion below -- the document parse itself,
    // the L/o/g/G, numeric and string fields, and the collective
    // implementation table -- shares one fail-closed channel: nlohmann
    // raises std::exception-derived type errors on a mismatched field type
    // (type_error.302), and initialize_sys runs inside the Sys constructor
    // where nothing above catches, so a bare throw used to terminate the
    // process without a message. Any read failure routes through the
    // existing sys_panic (critical log + exit(1)); a valid configuration
    // follows exactly the same path as before.
    try {
        inFile >> j;
        peak_perf_key_present = j.contains("peak-perf");
        local_mem_bw_key_present = j.contains("local-mem-bw");
        hbm_contention_key_present =
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
                sys_panic(
                    "unknown value for scheduling policy in sys input file");
            }
        }
        if (j.contains("inter-dimension-scheduling")) {
            string inp_inter_dimension_scheduling =
                j["inter-dimension-scheduling"];
            if (inp_inter_dimension_scheduling == "Ascending") {
                inter_dimension_scheduling =
                    InterDimensionScheduling::Ascending;
            } else if (inp_inter_dimension_scheduling == "RoundRobin") {
                inter_dimension_scheduling =
                    InterDimensionScheduling::RoundRobin;
            } else if (inp_inter_dimension_scheduling == "OfflineGreedy") {
                inter_dimension_scheduling =
                    InterDimensionScheduling::OfflineGreedy;
            } else {
                sys_panic("unknown value for inter-dimension-scheduling in "
                          "sys "
                          "input file");
            }
        }
        if (j.contains("collective-optimization")) {
            string inp_collective_optimization =
                j["collective-optimization"];
            if (inp_collective_optimization == "baseline") {
                collectiveOptimization = CollectiveOptimization::Baseline;
            } else if (inp_collective_optimization == "localBWAware") {
                collectiveOptimization = CollectiveOptimization::LocalBWAware;
            } else {
                sys_panic(
                    "unknown value for collective optimization in sys input "
                    "file");
            }
        }
        if (j.contains("local-reduction-delay")) {
            // 负值域守卫（2026-10-01 对齐）：LogGP 的 Processing_Finished
            // 延迟按 Tick（uint64）增量直接消费该值，负字面量会回绕成
            // ~1.8e19 ns、无任何点名本键的消息挂死 run——与下方
            // local-mem-latency 记录的同一失败族。
            const auto& lrd_value = j["local-reduction-delay"];
            if (!lrd_value.is_number_integer() &&
                !lrd_value.is_number_unsigned()) {
                sys_panic("local-reduction-delay must be a non-negative "
                          "integer number of ns in '" +
                          name + "'");
            }
            const int64_t parsed_lrd = lrd_value.get<int64_t>();
            if (parsed_lrd < 0 ||
                parsed_lrd > static_cast<int64_t>(
                                 std::numeric_limits<int>::max())) {
                sys_panic("local-reduction-delay out of domain "
                          "[0, int-max] in '" +
                          name + "' (got " + lrd_value.dump() + ")");
            }
            local_reduction_delay = static_cast<int>(parsed_lrd);
        }
        if (j.contains("active-chunks-per-dimension")) {
            // A 0/negative (or fractional/oversized) value would drive the
            // concurrent_streams computation to 0 (or wrap), so no stream is
            // ever initialized and the simulation silently hangs with zero
            // diagnosis. Fail closed at the read site, naming the key.
            const auto& chunks_value = j["active-chunks-per-dimension"];
            if (!chunks_value.is_number_integer() &&
                !chunks_value.is_number_unsigned()) {
                sys_panic("active-chunks-per-dimension must be an integer in "
                          "the system config");
            }
            const int64_t chunks = chunks_value.get<int64_t>();
            if (chunks < 1 ||
                chunks > static_cast<int64_t>(std::numeric_limits<int>::max())) {
                sys_panic("active-chunks-per-dimension must be in [1, int-max] "
                          "(got " + chunks_value.dump() + ")");
            }
            active_chunks_per_dimension = static_cast<int>(chunks);
        }
        // LogGP 整数域助手（2026-10-01 对齐）：MemBus/LogGP 的 L/o/g 构造
        // 参数是 Tick（无符号整型纳秒），而本解析器以 float 持有——小数
        // 配置值（如 "L": 0.5）过去在 memBus 构造点被静默截断为 0、零诊
        // 断。在边界拒绝小数值，而不是改下游整型（那会改变一切既有整数
        // 输入的计时路径算术，无收益）。
        const auto require_integral_ns = [&name](const double value,
                                                 const char* key) {
            if (value != std::floor(value)) {
                sys_panic(std::string(key) +
                          " must be a whole number of nanoseconds in '" +
                          name +
                          "' (fractional LogGP latencies are truncated by "
                          "the Tick-typed bus parameters; got " +
                          std::to_string(value) + ")");
            }
        };
        if (j.contains("L")) {
            // 严格解析（与下方 endpoint-delay 同族）：非数字字面量过去
            // 只以不点名本键的泛型 json 类型错误冒出，负值在 MemBus 转
            // Tick 时回绕。
            const auto& l_value = j["L"];
            if (!l_value.is_number() ||
                !std::isfinite(l_value.get<double>()) ||
                l_value.get<double>() < 0.0) {
                sys_panic("L must be a finite non-negative number in '" +
                          name + "'");
            }
            require_integral_ns(l_value.get<double>(), "L");
            inp_L = l_value.get<float>();
        }
        if (j.contains("o")) {
            const auto& o_value = j["o"];
            if (!o_value.is_number() ||
                !std::isfinite(o_value.get<double>()) ||
                o_value.get<double>() < 0.0) {
                sys_panic("o must be a finite non-negative number in '" +
                          name + "'");
            }
            require_integral_ns(o_value.get<double>(), "o");
            inp_o = o_value.get<float>();
        }
        if (j.contains("g")) {
            const auto& g_value = j["g"];
            if (!g_value.is_number() ||
                !std::isfinite(g_value.get<double>()) ||
                g_value.get<double>() < 0.0) {
                sys_panic("g must be a finite non-negative number in '" +
                          name + "'");
            }
            require_integral_ns(g_value.get<double>(), "g");
            inp_g = g_value.get<float>();
        }
        if (j.contains("G")) {
            const auto& G_value = j["G"];
            if (!G_value.is_number() ||
                !std::isfinite(G_value.get<double>()) ||
                G_value.get<double>() < 0.0) {
                sys_panic("G must be a finite non-negative number in '" +
                          name + "'");
            }
            inp_G = G_value.get<float>();
        }
        if (j.contains("endpoint-delay")) {
            // injection-scale 均匀拉伸端点时延：在 double 域读 JSON 值并
            // 一次性取整（JSON 3.5 × scale 2 旧的 double→int 截断丢
            // 0.5 ns/消息）。键缺席的缺省（10）同样乘 scale——缩放 run
            // 不再在一个进程里混用两种约定。
            // 负值域守卫（2026-10-01 对齐）：MemBus 把缩放后的值按
            // Tick（uint64）增量注册，负结果会回绕成 ~1.8e19 ns、无点
            // 名本键的消息挂死 run。
            const auto& delay_value = j["endpoint-delay"];
            if (!delay_value.is_number()) {
                sys_panic("endpoint-delay must be a finite non-negative "
                          "number of ns in '" +
                          name + "'");
            }
            const double scaled =
                delay_value.get<double>() * injection_scale;
            if (!std::isfinite(scaled) || scaled < 0.0 ||
                scaled >
                    static_cast<double>(std::numeric_limits<int>::max())) {
                sys_panic("endpoint-delay out of domain [0, int-max] ns "
                          "after injection scaling in '" +
                          name + "' (scaled value " +
                          std::to_string(scaled) + ")");
            }
            communication_delay = static_cast<int>(std::llround(scaled));
        } else {
            // 键缺席的缺省（10）走同一缩放与同一值域门（R2 复核补强）：
            // injection_scale 无上界（CmdLineParser 只查有限/正），巨值
            // 缩放下 10*scale 溢为 inf、llround(inf) 即 UB——与显式分支
            // 同款三重拒绝。
            const double default_scaled = 10 * injection_scale;
            if (!std::isfinite(default_scaled) || default_scaled < 0.0 ||
                default_scaled >
                    static_cast<double>(
                        std::numeric_limits<int>::max())) {
                sys_panic("endpoint-delay default (10) out of domain "
                          "[0, int-max] ns after injection scaling in '" +
                          name + "' (scaled value " +
                          std::to_string(default_scaled) + ")");
            }
            communication_delay =
                static_cast<int>(std::llround(default_scaled));
        }
        if (j.contains("model-shared-bus")) {
            // 布尔/整型严格解析（与 roofline-enabled、
            // hbm-bandwidth-contention 同通道）：旧 `== 1` 比较把其它任
            // 何非零整数（如 2，语义上是"开"）静默读成关——fail-open 且
            // 无诊断。
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
            // 严格类型/值域解析：该值进 determine_chunk_size 的除法，非
            // 正或小数字面量过去静默落入该函数的单 chunk 回退、无点
            // 名本键的诊断。
            const auto& splits_value = j["preferred-dataset-splits"];
            if (!splits_value.is_number_integer() &&
                !splits_value.is_number_unsigned()) {
                sys_panic("preferred-dataset-splits must be an integer in "
                          "'" +
                          name + "'");
            }
            const int64_t parsed_splits = splits_value.get<int64_t>();
            if (parsed_splits < 1 ||
                parsed_splits >
                    static_cast<int64_t>(std::numeric_limits<int>::max())) {
                sys_panic("preferred-dataset-splits out of domain "
                          "[1, int-max] in '" +
                          name + "' (got " + splits_value.dump() + ")");
            }
            preferred_dataset_splits = static_cast<int>(parsed_splits);
        }
        if (j.contains("peak-perf")) {
            // 值域守卫（2026-10-01 对齐）：peak_perf 在计算计时路径上做
            // 除数，显式非正值会构造出 get_perf <= 0 的 Roofline——下游
            // 除法产出负/inf 运行时长且无点名本键的诊断。workload 层对
            // 键缺席的缺省 0 保留自己的用点守卫；显式配置的非正
            // peak-perf 在此边界拒绝。
            const auto& peak_value = j["peak-perf"];
            if (!peak_value.is_number()) {
                sys_panic("peak-perf must be a positive number of TFLOPS "
                          "in '" +
                          name + "'");
            }
            const double scaled_peak = peak_value.get<double>() * 1000000000000;
            if (!std::isfinite(scaled_peak) || scaled_peak <= 0.0) {
                sys_panic("peak-perf must be positive (TFLOPS) in '" +
                          name + "' (got " + peak_value.dump() + ")");
            }
            peak_perf = scaled_peak;  // TFLOPS
        }
        if (j.contains("local-mem-bw")) {
            // 值域守卫（2026-10-01 对齐），与 peak-perf 对称：闭式
            // roofline 臂除以 local_mem_bw，病态小正值（如 1e-300 GB/s）
            // 能活过下方"<= 0 关流体模型"检查、却在闭式 cast 把
            // elapsed_time 溢出成 inf；NaN 藏在每个 "<= 0" 比较后面。
            // 在边界以键名拒绝非有限与非正值。
            const auto& bw_value = j["local-mem-bw"];
            if (!bw_value.is_number()) {
                sys_panic("local-mem-bw must be a positive number of "
                          "GB/sec in '" +
                          name + "'");
            }
            const double scaled_bw = bw_value.get<double>() * 1000000000;
            if (!std::isfinite(scaled_bw) || scaled_bw <= 0.0) {
                sys_panic("local-mem-bw must be positive (GB/sec) in '" +
                          name + "' (got " + bw_value.dump() + ")");
            }
            local_mem_bw = scaled_bw;  // GB/sec -> B/sec
        }
        if (j.contains("local-mem-latency")) {
            // 严格时延解析（与相邻字段同一 panic 纪律）：nlohmann 按词
            // 法转换 JSON 值，负字面量（-100）重解释为 uint64_t 变成
            // ~1.8e19 ns——roofline 闭式路径用它算 now+runtime、回绕事件
            // 时钟、触发 EventQueue 严格递增中止（其消息不点名本字
            // 段）；流体 HBM 路径上同一值把传输事件重排进过去、直到看
            // 门狗杀 run。非整数或值域外必须在此边界点名失败。
            const auto& latency_value = j["local-mem-latency"];
            if (!latency_value.is_number_integer() &&
                !latency_value.is_number_unsigned()) {
                sys_panic("local-mem-latency must be a non-negative "
                          "integer number of ns in '" +
                          name + "'");
            }
            const int64_t parsed_latency = latency_value.get<int64_t>();
            if (parsed_latency < 0 ||
                parsed_latency > (static_cast<int64_t>(1) << 62)) {
                sys_panic("local-mem-latency out of domain [0, 2^62] ns "
                          "(got " +
                          std::to_string(parsed_latency) + " in '" + name +
                          "')");
            }
            local_mem_latency = static_cast<uint64_t>(parsed_latency);
        }
        if (j.contains("remote-mem-bw")) {
            // 值域守卫（2026-10-01，与 local-mem-bw 同族；本键为 LRU 仓
            // 特有）：NaN 藏在每个 "<= 0" 比较后面、病态小正值在下游
            // 除法里溢出成 inf——在边界以键名拒绝非有限与非正值。
            const auto& remote_bw_value = j["remote-mem-bw"];
            if (!remote_bw_value.is_number()) {
                sys_panic("remote-mem-bw must be a positive number of "
                          "GB/sec in '" +
                          name + "'");
            }
            const double scaled_remote_bw =
                remote_bw_value.get<double>() * 1000000000;
            if (!std::isfinite(scaled_remote_bw) ||
                scaled_remote_bw <= 0.0) {
                sys_panic("remote-mem-bw must be positive (GB/sec) in '" +
                          name + "' (got " + remote_bw_value.dump() + ")");
            }
            remote_mem_bw = scaled_remote_bw;  // GB/sec -> B/sec
            // Unit convention (RB-01, 2026-10-01): this system.json key is
            // written by config_resolver.py as a decimal GB/s value and is
            // consumed HERE as GB/s (x1e9 -> B/s). The remote_memory.json
            // key of the SAME NAME carries the same source number but is
            // consumed by AnalyticalRemoteMemory.cc as a RAW bytes/ns
            // value (no conversion). The identity GB/s == B/ns (10^9
            // decimal) is what keeps both consumers correct off one source
            // value -- the two generated files must never be edited
            // independently.
        }
        if (j.contains("remote-mem-latency")) {
            // 严格时延解析（与 local-mem-latency 同族；本键为 LRU 仓特
            // 有）：负字面量按 uint64_t 重解释回绕成 ~1.8e19 ns、把远端
            // 端口变迁事件重排进过去直到看门狗杀 run；非整数或值域外
            // 在此边界点名失败。
            const auto& remote_latency_value = j["remote-mem-latency"];
            if (!remote_latency_value.is_number_integer() &&
                !remote_latency_value.is_number_unsigned()) {
                sys_panic("remote-mem-latency must be a non-negative "
                          "integer number of ns in '" +
                          name + "'");
            }
            const int64_t parsed_remote_latency =
                remote_latency_value.get<int64_t>();
            if (parsed_remote_latency < 0 ||
                parsed_remote_latency > (static_cast<int64_t>(1) << 62)) {
                sys_panic("remote-mem-latency out of domain [0, 2^62] ns "
                          "(got " +
                          std::to_string(parsed_remote_latency) + " in '" +
                          name + "')");
            }
            remote_mem_latency =
                static_cast<uint64_t>(parsed_remote_latency);  // ns
        }
        if (j.contains("pipeline-tile-fraction")) {
            pipeline_tile_fraction = j["pipeline-tile-fraction"];
            pipeline_tile_fraction =
                std::max(0.0, std::min(1.0, pipeline_tile_fraction));
        }
        if (j.contains("roofline-enabled")) {
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
        if (j.contains("hbm-kv-restore-bandwidth-sharing")) {
            const auto& sharing = j["hbm-kv-restore-bandwidth-sharing"];
            if (sharing.is_boolean()) {
                hbm_kv_restore_bandwidth_sharing = sharing.get<bool>();
            } else if (sharing.is_number_integer() ||
                       sharing.is_number_unsigned()) {
                hbm_kv_restore_bandwidth_sharing =
                    sharing.get<int64_t>() != 0;
            } else {
                sys_panic(
                    "hbm-kv-restore-bandwidth-sharing must be boolean or "
                    "integer");
            }
        }
        // Multi-user local-HBM bandwidth contention
        // (hbm-bandwidth-contention, bool/int, default true). Parsed after
        // local-mem-bw so the zero-rate guard below sees the final
        // bandwidth value.
        if (j.contains("hbm-bandwidth-contention")) {
            const auto& contention = j["hbm-bandwidth-contention"];
            if (contention.is_boolean()) {
                hbm_bandwidth_contention = contention.get<bool>();
            } else if (contention.is_number_integer() ||
                       contention.is_number_unsigned()) {
                hbm_bandwidth_contention =
                    contention.get<int64_t>() != 0;
            } else {
                sys_panic(
                    "hbm-bandwidth-contention must be boolean or integer");
            }
        }
        if (local_mem_bw <= 0) {
            // Zero-rate guard: a fluid model over a zero local-mem-bw would
            // never move a byte; fall back to the legacy behavior instead.
            hbm_bandwidth_contention = false;
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
        // "replay-only"：随 Path-2 清除（2026-10-01）已无消费方——
        // Workload::issue 的整节点 replay 分派已删除，键仅为模板兼容保留
        // 在白名单中（known_system_keys）并被解析成成员，任何取值都不再
        // 驱动行为。
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
        // "track-local-mem" / "local-mem-trace-filename" are retired keys:
        // the ETFeeder-bound LocalMemUsageTracker chain they fed was removed
        // with the static path. They are deliberately NOT in
        // known_system_keys below, so a configuration still carrying them
        // now fails closed in the unknown-key gate instead of being
        // silently ignored.

        // Unknown-key gate (3.12, ported from the wscllm mother repo,
        // 2026-10-01): this reader only consumed known keys and silently
        // ignored everything else, so a typo'd template key
        // ("endpoint-dely", "roofline-enable") fell back to the built-in
        // default and the feature silently turned off with zero
        // diagnostics -- fail-open at the last line of defense. The
        // hardware json side already enforces an exact-key whitelist
        // (_require_exact_keys in the resolver); this is the matching gate
        // for the system side. Unlike the mother repo -- where the custom
        // path is retired -- this tree keeps custom collectives live, so
        // the five custom keys below are REAL consumed keys (they are read
        // by setup_collective_impl_from_config right after this gate), not
        // retirement-diagnosis entries. "intra-dimension-scheduling" (known
        // to the mother repo) is deliberately absent: this repo has no
        // reader for it, so carrying it must stay a loud error rather than
        // a silently ignored key. Keep this set in sync with
        // _KNOWN_SYSTEM_TEMPLATE_KEYS in sh_test_mesh/config_resolver.py
        // (the earliest boundary, same contract).
        static const std::set<std::string> known_system_keys = {
            "scheduling-policy",
            "inter-dimension-scheduling",
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
            "roofline-enabled",
            "hbm-kv-restore-bandwidth-sharing",
            "hbm-bandwidth-contention",
            "trace-enabled",
            "replay-only",
            "compute-cost-model",
            "die-model-profile",
            "die-model-config",
            "all-reduce-implementation",
            "reduce-scatter-implementation",
            "all-gather-implementation",
            "all-to-all-implementation",
            // custom collectives are LIVE in this tree: consumed by
            // setup_collective_impl_from_config below.
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
                      " (a typo'd key used to silently fall back to the "
                      "default value and disable the intended feature)");
        }
        collective_impl_lookup->setup_collective_impl_from_config(j);
    } catch (const std::exception& e) {
        sys_panic(std::string("failed to read system configuration '") +
                  name + "': " + e.what());
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
        // designed instead of silently running the contention-off arm
        // (this matters for this repository: the N-way arbiter also serves
        // RESTORE / POOL_READ / POOL_WRITE / comm endpoints).
        if (!hbm_contention_key_present && local_mem_bw > 0) {
            hbm_bandwidth_contention = true;
        }
    }

    die_runtime = std::make_unique<DieModelRuntime>(std::move(hw));
}

Sys::DieModelRuntime::DieModelRuntime(DieModel::Hardware hww)
    : hardware(std::move(hww)), plan_cache(shared_die_plan_cache(hardware)) {}

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

// The name is historical: the simulation loop itself is owned by the network
// backend's event loop. This only logs the final teardown message emitted
// when the last Sys object is destroyed; it does not exit any loop.
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
            // Fail closed: an exception here leaves the stream / DataSet /
            // metric state at an arbitrary midpoint of the callback, and
            // consuming the event as if it had succeeded would let the
            // simulation run on silently corrupt. Paths that must never
            // throw fail with critical + exit at the source; a throw
            // reaching this catch is a bug and stops the run with a
            // diagnosis. (Online is the only constructible execution mode:
            // the former Static swallow-and-continue branch was dead -- the
            // mode default is Online, Workload fail-closes any non-Online
            // construction, and no producer ever passed Static.)
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
    // 深度审查 M-2（P2-10，2026-09-30，移植自姊妹仓 face-LRU 同款 H3 修复）：
    // delta==0 的注册若发生在后端 invoke 派发轮之外，会在事件队列同刻再开
    // 一桶——严格递增守卫（invoke 轮之外禁止同刻 schedule_event）直接终止
    // 整跑。在线 tick-end / deferred 提交会走本路径发射集合通信，且
    // communication_delay 可被合法配置折叠为 0（如分数 injection-scale 截
    // 断），首个集合通信包即以一条与根因无关的报错致死。invoke 派发轮内
    // 的同刻合并语义保持逐字节不变，仅禁止上下文推迟到下一 tick。
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
        // 不变量：每个载荷至多武装一个消费者（workload 臂在 call() 内
        // 释放 wlhd，第二个武装的消费者即 use-after-free；custom_algorithm
        // 臂同理）。fail-closed 而不是放行。
        const int armed = (rcehd->workload != nullptr) +
                          (rcehd->owner != nullptr) +
                          (rcehd->custom_algorithm != nullptr);
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
        if (rcehd->custom_algorithm) {
            rcehd->custom_algorithm->call(event, rcehd->wlhd);
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
        // undefined behavior. call_events has already consumed the event at
        // this point, so continuing would let the run proceed from a silently
        // corrupt event state (the payload is deliberately leaked here);
        // fail closed instead.
        LoggerFactory::get_logger("system")->critical(
            "Sys::handleEvent: unknown event type {} on sys {}; "
            "payload not freed; failing closed",
            static_cast<int>(event), id);
        std::exit(EXIT_FAILURE);
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
    // The custom-collective path below does not consume chunk_size at all;
    // for every other path a missing/non-positive 'preferred-dataset-splits'
    // key falls back to a whole-block chunk in determine_chunk_size, so the
    // value no longer has to be present in the system JSON input.
    // Static-mode counterpart of the online commit-boundary preflight: a
    // zero-size collective cannot be split into streams and a zero chunk
    // size would divide the stream count below (NaN-to-int / zero-stream
    // DataSet), so both are rejected here instead of producing an
    // inconsistent execution.
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
            // Below three are default values.
            RingTopology::Direction::Clockwise,
            implementation_per_dimension[0],
            communicator_group);
        list<CollectivePhase> vect;
        vect.push_back(phase);
        // 流 id 耗尽守卫（2026-10-01 对齐）：集合流 id 兼作 COLLECTIVE tag，
        // 组窗口 = [id*1e6, (id+1)*1e6)。触达上界仍递增会把流 id 溢进
        // 下一组的 tag 块（经 %5e8 重映射后跨组 (src,dst,tag) 别名 = 错
        // 配对或永久互等）——在分配前检查并 fail-closed。
        int stream_id = -1;
        if (communicator_group != nullptr) {
            if (communicator_group->num_streams >=
                communicator_group->stream_tag_ceiling()) {
                sys_panic("communicator group " +
                          std::to_string(communicator_group->get_id()) +
                          " exhausted its collective stream id window");
            }
            stream_id = communicator_group->num_streams++;
        } else {
            // 世界组窗口 [0, 1e6)；Sys::num_streams 只在世界分支递增。
            if (num_streams >= 1000000) {
                sys_panic("world communicator group exhausted its "
                          "collective stream id window");
            }
            stream_id = num_streams++;
        }
        StreamBaseline* newStream =
            new StreamBaseline(this, dataset, stream_id, vect, pri);
        newStream->current_queue_id = -1;
        insert_into_ready_list(newStream);
        // Exactly one stream is created here while the dataset above was
        // pre-sized with the chunk math; align it so the dataset can reach
        // finished_streams == total_streams instead of waiting forever.
        dataset->total_streams = 1;
        return dataset;
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
                inter_dimension_scheduling, collective_type);
            chunk_size = prev_size - size;
        }

        // P2 对齐（2026-10-01，母仓 2026-09-28 已落地）：splits 不可整除的
        // 集合通信其尾段流必须携带**实际剩余字节**。旧的无条件
        // remain_size = chunk_size 把尾段补齐到整 chunk_size，系统性虚增
        // 集合通信流量（All_Gather 同一赋值路径）。流数（ceil）不变。
        // 原 sys-core-E4（2026-09-27）裁决"先要在每个集合算法的字节
        // 算术里加地板守卫才能修"的前提已失效——DB1 1 字节下限
        // （Ring.cc 构造器，AllToAll 子类共享）与 HalvingDoubling 的
        // 中途折半/转折点 clamp 均已随母仓移植在仓，小于环节点数的
        // 尾余量不再整除出零字节 ring step。
        if (collective_type == ComType::All_to_All ||
            inter_dimension_scheduling !=
                InterDimensionScheduling::OfflineGreedy) {
            if (chunk_size > size) {
                remain_size = size;  // 尾段：实际剩余字节
                size = 0;
            } else {
                remain_size = chunk_size;
                size -= chunk_size;
            }
        } else {
            // OfflineGreedy（非 All_to_All）：上方 chunk_size 已被改写为
            // 本次迭代实际消耗的字节数，本身就是尾段精确余量。
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
            // 流 id 耗尽守卫（2026-10-01 对齐）：同上——分配前检查窗口
            // 上界，触达即 fail-closed，防止静默溢进下一组 tag 块。
            int stream_id = -1;
            if (communicator_group != nullptr) {
                if (communicator_group->num_streams >=
                    communicator_group->stream_tag_ceiling()) {
                    sys_panic("communicator group " +
                              std::to_string(communicator_group->get_id()) +
                              " exhausted its collective stream id window");
                }
                stream_id = communicator_group->num_streams++;
            } else {
                if (num_streams >= 1000000) {
                    sys_panic("world communicator group exhausted its "
                              "collective stream id window");
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
            // it through the event loop instead: one self event acts as the
            // only "stream", so when it pops, notify_stream_finished fires
            // the notifier that the workload layer registers right after
            // this generate_* call returns. (This local-completion pattern
            // is unrelated to the workload's BROADCAST comm type, which is
            // fail-closed rejected in Workload's issue path -- only the
            // degenerate-dimension shape lands here.)
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
        // 误配对（DBT 实现配非树拓扑）必须大声失败，而不是经 C 式下行
        // 转型读垃圾。
        BinaryTree* btree = dynamic_cast<BinaryTree*>(topology);
        if (btree == nullptr) {
            sys_panic("DoubleBinaryTree requires a BinaryTree logical "
                      "topology");
        }
        // F-RV8-1（2026-09-29 母仓同款）：DoubleBinaryTreeAllReduce 硬编码
        // All_Reduce 语义（算法类无 ComType 参数），all-gather/
        // reduce-scatter-implementation 键下配 doubleBinaryTree 会静默按
        // 错误的集合语义仿真且照样"完成"——与 Ring::run 里 plain-Ring 拒
        // 驱动 All_to_All 同族。在相位生成处 fail-closed。
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
                           new DoubleBinaryTreeAllReduce(
                               id, btree, data_size));
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
        CollectivePhase vn(0, new CustomAlgorithm(filename, queue_id, comm_group));
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
    // Intra-dimension scheduling is pinned to FIFO: streams enter the queue
    // ahead of not-yet-initialized streams with lower priority.
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
    if (stream->my_current_phase.algorithm != nullptr) {
        delete stream->my_current_phase.algorithm;
    }
    if (stream->phases_to_go.size() == 0) {
        stream->dataset->notify_stream_finished();
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
