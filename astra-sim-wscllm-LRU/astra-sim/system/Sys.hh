/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __SYSTEM_HH__
#define __SYSTEM_HH__

#include <memory>
#include <string>

#include "astra-sim/common/AstraNetworkAPI.hh"
#include "astra-sim/common/AstraRemoteMemoryAPI.hh"
#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/CollectivePhase.hh"
#include "astra-sim/system/CommunicatorGroup.hh"
#include "astra-sim/system/MemBus.hh"
#include "astra-sim/system/Roofline.hh"
#include "astra-sim/system/astraccl/CollectiveImplLookup.hh"
#include "astra-sim/system/astraccl/native_collectives/logical_topology/RingTopology.hh"
#include "astra-sim/workload/Workload.hh"
#include "DieTypes.hh"

namespace AstraSim {

class BaseStream;
class StreamBaseline;
class DataSet;
class QueueLevels;
class Workload;
class LogicalTopology;
class BasicLogicalTopology;
class OfflineGreedy;

// Compute cost backend selector (die内仿真器 stage B; design doc §8.2,
// contract §3.2). The system.json key "compute-cost-model" carries the
// string spelling ("die" | "roofline" | "replay"); absent key keeps
// LegacyDefault, which -- for backward compatibility -- still means the
// pre-stage-B dispatch (roofline only when roofline-enabled=1).
enum class ComputeCostModel { LegacyDefault, Roofline, Replay, Die };

namespace ExecutionDriven {
class GraphSource;
}  // namespace ExecutionDriven

class Sys : public Callable {
  public:
    // SchedulerUnit
    // ------------------------------------------------------------
    class SchedulerUnit {
      public:
        SchedulerUnit(Sys* sys,
                      std::vector<int> queues,
                      int max_running_streams,
                      int ready_list_threshold,
                      int queue_threshold);
        void notify_stream_added(int vnet);
        void notify_stream_added_into_ready_list();
        void notify_stream_removed(int vnet);

        Sys* sys;
        int max_running_streams;
        int ready_list_threshold;
        int queue_threshold;
        std::map<int, int> running_streams;
        std::map<int, std::list<BaseStream*>::iterator> stream_pointer;
    };
    //---------------------------------------------------------------------------

    // Constructor / Destructor
    // -------------------------------------------------
    // execution_mode/graph_source: step-1-2 execution-mode factory
    // (ExecutionMode.hh / GraphSource.hh). The offline static path is
    // retired in this repository (the ETFeederGraphSource adapter was
    // removed with it); Online is the only constructible mode: it never
    // constructs the ETFeeder and never requires .et files.
    Sys(int id,
        std::string workload_configuration,
        std::string comm_group_configuration,
        std::string system_configuration,
        AstraRemoteMemoryAPI* remote_mem,
        AstraNetworkAPI* comm_NI,
        std::vector<int> physical_dims,
        std::vector<int> queues_per_dim,
        double injection_scale,
        double comm_scale,
        bool rendezvous_enabled,
        ExecutionDriven::ExecutionMode execution_mode =
            ExecutionDriven::ExecutionMode::Online,
        std::shared_ptr<ExecutionDriven::GraphSource> graph_source = nullptr);
    ~Sys();
    //---------------------------------------------------------------------------

    // Intialization
    // ------------------------------------------------------------
    bool initialize_sys(std::string name);

  private:
    // Stage B (die backend; contract §3.2/§3.3): loads + validates the
    // resolved die model config through DieModel::make_hardware, enforces
    // the derived-scalar consistency of peak-perf/local-mem-bw, fills the
    // missing ones from the library, and installs die_runtime. Every
    // failure path exits (fail-closed). Called from initialize_sys after
    // the whole key-walk, only for an explicit compute-cost-model=die.
    void initialize_die_backend(const std::string& system_config_path,
                                bool explicit_config_present,
                                const std::string& explicit_config_path,
                                bool profile_present,
                                const std::string& profile,
                                bool peak_perf_key_present,
                                bool local_mem_bw_key_present,
                                bool hbm_contention_key_present);

  public:
    //---------------------------------------------------------------------------

    // Helper Functions
    // ---------------------------------------------------------
    static Tick boostedTick();
    static void sys_panic(std::string msg);
    //---------------------------------------------------------------------------

    // Simulation Loop
    // ----------------------------------------------------------
    void exit_sim_loop(std::string msg);
    //---------------------------------------------------------------------------

    // General Event Handling
    // ---------------------------------------------------
    void call(EventType type, CallData* data);
    void call_events();
    void register_event(Callable* callable,
                        EventType event,
                        CallData* callData,
                        Tick delta_cycles);
    // Cancellable events own neither Callable nor CallData by default.  A
    // caller that supplies a cancellation callback transfers only the
    // pending-payload cleanup to Sys; after the event is popped, its normal
    // Callable::call path keeps the pre-existing ownership contract.
    using EventDataCancellationCallback = void (*)(CallData*);
    [[nodiscard]] SystemEventHandle register_event_cancellable(
        Callable* callable,
        EventType event,
        CallData* callData,
        Tick delta_cycles,
        EventDataCancellationCallback cancellation_callback);
    [[nodiscard]] bool cancel_event(SystemEventHandle& handle);
    void try_register_event(Callable* callable,
                            EventType event,
                            CallData* callData,
                            Tick& delta_cycles);
    static void handleEvent(void* arg);
    //---------------------------------------------------------------------------

    // Communicator Group Support
    // -----------------------------------------------
    LogicalTopology* get_logical_topology(ComType comm_type);
    //---------------------------------------------------------------------------

    // Collective Communication Primitives
    // --------------------------------------

    // [operation specific custom collective]
    // We want to designate different collective algorithms for different collective operations.
    // To do that, when determining which collective algorithm to use, the system layer needs to know
    // which operator (i.e. Chakra node) it is trying to simulate (so that it can look up the algorithm)
    // Therefore, we provide the id of the Chakra node to use as a lookup key.
    DataSet* generate_all_reduce(uint64_t size,
                                 std::vector<bool> involved_dimensions,
                                 CommunicatorGroup* communicator_group,
                                 int explicit_priority,
                                 uint64_t workload_node_id = -1);
    DataSet* generate_all_to_all(uint64_t size,
                                 std::vector<bool> involved_dimensions,
                                 CommunicatorGroup* communicator_group,
                                 int explicit_priority,
                                 uint64_t workload_node_id = -1);
    DataSet* generate_all_gather(uint64_t size,
                                 std::vector<bool> involved_dimensions,
                                 CommunicatorGroup* communicator_group,
                                 int explicit_priority,
                                 uint64_t workload_node_id = -1);
    DataSet* generate_reduce_scatter(uint64_t size,
                                     std::vector<bool> involved_dimensions,
                                     CommunicatorGroup* communicator_group,
                                     int explicit_priority,
                                     uint64_t workload_node_id = -1);
    DataSet* generate_collective(
        uint64_t size,
        LogicalTopology* topology,
        std::vector<CollectiveImpl*> implementation_per_dimension,
        std::vector<bool> dimensions_involved,
        ComType collective_type,
        int explicit_priority,
        CommunicatorGroup* communicator_group);
    CollectivePhase generate_collective_phase(ComType collective_type,
                                              BasicLogicalTopology* topology,
                                              uint64_t data_size,
                                              int queue_id,
                                              RingTopology::Direction direction,
                                              CollectiveImpl* collective_impl,
                                              CommunicatorGroup* comm_group = nullptr);
    //---------------------------------------------------------------------------

    // Middle-level Network Primitives
    // ------------------------------------------
    uint64_t determine_chunk_size(uint64_t& size, ComType type);
    int get_priority(int explicit_priority);
    void insert_into_ready_list(BaseStream* stream);
    void insert_stream(std::list<BaseStream*>* queue, BaseStream* baseStream);
    void schedule(int num);
    void proceed_to_next_vnet_baseline(StreamBaseline* stream);
    //---------------------------------------------------------------------------

    // Low-level Network Primitives
    // ---------------------------------------------
    enum FrontEndSendRecvType {
        // NATIVE means send/recv is issued directly by workload input
        // COLLECTIVE means send/recv is caused by a collective communication
        // RENDEZVOUS means send/recv is a rendezvous shake hand
        // The value here presents the offset of the tag. The tag range for
        // different type is as follows
        // NATIVE: [0, 500000000)
        // COLLECTIVE: [500000000, 1000000000)
        // RENDEZVOUS: [1000000000, 2000000000)
        NATIVE = 0,
        COLLECTIVE = 500000000,
        RENDEZVOUS = 1000000000
    };
    int front_end_sim_send(Tick delay,
                           void* buffer,
                           uint64_t count,
                           int type,
                           int dst,
                           int tag,
                           sim_request* request,
                           FrontEndSendRecvType send_type,
                           void (*msg_handler)(void* fun_arg),
                           void* fun_arg);

    int front_end_sim_recv(Tick delay,
                           void* buffer,
                           uint64_t count,
                           int type,
                           int src,
                           int tag,
                           sim_request* request,
                           FrontEndSendRecvType recv_type,
                           void (*msg_handler)(void* fun_arg),
                           void* fun_arg);

    int rendezvous_sim_send(Tick delay,
                            void* buffer,
                            uint64_t count,
                            int type,
                            int dst,
                            int tag,
                            sim_request* request,
                            void (*msg_handler)(void* fun_arg),
                            void* fun_arg);

    int rendezvous_sim_recv(Tick delay,
                            void* buffer,
                            uint64_t count,
                            int type,
                            int src,
                            int tag,
                            sim_request* request,
                            void (*msg_handler)(void* fun_arg),
                            void* fun_arg);

    int sim_send(Tick delay,
                 void* buffer,
                 uint64_t count,
                 int type,
                 int dst,
                 int tag,
                 sim_request* request,
                 void (*msg_handler)(void* fun_arg),
                 void* fun_arg);

    int sim_recv(Tick delay,
                 void* buffer,
                 uint64_t count,
                 int type,
                 int src,
                 int tag,
                 sim_request* request,
                 void (*msg_handler)(void* fun_arg),
                 void* fun_arg);
    //---------------------------------------------------------------------------

    static std::vector<Sys*> all_sys;  // vector of all Sys objects

    int id;

    // workload
    Workload* workload;

    // step-1-2 execution-mode factory state (see constructor comment).
    ExecutionDriven::ExecutionMode execution_mode_ =
        ExecutionDriven::ExecutionMode::Online;
    std::shared_ptr<ExecutionDriven::GraphSource> graph_source_ = nullptr;
    // roofline model
    bool roofline_enabled;
    double peak_perf;
    Roofline* roofline;

    // Compute cost backend (stage B; see the ComputeCostModel enum above).
    // Parsed from the system.json "compute-cost-model" key with the contract
    // §3.2 conflict matrix enforced fail-closed in Sys::initialize_sys.
    ComputeCostModel compute_cost_model = ComputeCostModel::LegacyDefault;

    // Die backend runtime aggregate (stage B): the validated die hardware
    // (loaded from die-model-config, a resolved_die_model.json produced by
    // the resolver through the cost library) plus the plan cache (LRU 4096;
    // design doc §8.5 -- "执行句柄不入缓存", only immutable plans are
    // cached). Null unless compute_cost_model == Die. Declared AFTER the
    // workload pointer it feeds: the Workload (and its DieModelAdapter /
    // live DieExecutions holding shared_ptr<const DiePlan>) is destroyed in
    // ~Sys's body, before this member unwinds.
    //
    // Stage E2 (as delivered in the face reference): the PlanCache is
    // PROCESS-LEVEL SHARED across every Sys of the same resolved hardware
    // (one registry entry per plan_cache_hardware_key). compile() is a pure
    // function and a DiePlan is immutable, so ranks compiling the same
    // (spec, hardware) pair -- which always produces the same
    // plan_cache_key -- can safely reuse each other's plans; per-Sys caches
    // forced every Sys to recompile identical keys. Sharing does not change
    // hit semantics: the key already contains every distinguishing field.
    // The simulation is single-threaded (one event loop drives all Sys), so
    // the registry and the shared cache carry no lock; if a multi-threaded
    // runtime is ever introduced, guard both with a mutex.
    struct DieModelRuntime {
        DieModel::Hardware hardware;
        std::shared_ptr<DieModel::PlanCache> plan_cache;
        explicit DieModelRuntime(DieModel::Hardware hw);
    };
    std::unique_ptr<DieModelRuntime> die_runtime;
    // Resolved die-model-config path (diagnostics only; empty off the die
    // backend).
    std::string die_model_config_path;

    // memory
    double local_mem_bw;
    uint64_t local_mem_latency;
    double remote_mem_bw;
    uint64_t remote_mem_latency;
    double pipeline_tile_fraction;
    bool hbm_kv_restore_bandwidth_sharing;
    // N-way equal-split HBM contention (system key "hbm-bandwidth-contention",
    // code default true): true = the LocalHbmBandwidthModel takes over every
    // HBM user on the rank (COMP, KV restore, NoC p2p comm endpoints, pool
    // endpoints) under full_rate/N strict equal split; false = the legacy
    // behavior is fully preserved (the old "hbm-kv-restore-bandwidth-sharing"
    // two-user 50/50 semantics remain available for A/B). local-mem-bw <= 0
    // auto-disables the flag.
    bool hbm_bandwidth_contention;
    AstraRemoteMemoryAPI* remote_mem;

    // memory bus
    MemBus* memBus;
    float inp_L;
    float inp_o;
    float inp_g;
    float inp_G;
    bool model_shared_bus;
    double injection_scale;
    int communication_delay;
    int local_reduction_delay;

    // network
    AstraNetworkAPI* comm_NI;
    bool rendezvous_enabled;

    // scheduler
    SchedulerUnit* scheduler_unit;
    QueueLevels* vLevels;
    OfflineGreedy* offline_greedy;
    InterDimensionScheduling inter_dimension_scheduling;
    int round_robin_inter_dimension_scheduler;
    int active_chunks_per_dimension;
    int priority_counter;
    uint64_t pending_events;
    int preferred_dataset_splits;

    // Streams awaiting first schedule (all scheduling policies)
    std::list<BaseStream*> ready_list;
    SchedulingPolicy scheduling_policy;
    int first_phase_streams;
    int total_running_streams;
    std::map<int, std::list<BaseStream*>> active_Streams;

    struct ScheduledEvent {
        Callable* callable;
        EventType event;
        CallData* call_data;
        uint64_t event_id;
        EventDataCancellationCallback cancellation_callback;
    };
    struct ScheduledEventBucket {
        std::list<ScheduledEvent> events;
        AstraNetworkAPI::CancellableScheduleHandle outer_alarm;
    };
    std::map<Tick, ScheduledEventBucket> event_queue;
    uint64_t next_cancellable_event_id = 1;
    bool dispatching_events = false;
    Tick dispatching_event_time = 0;
    int total_nodes;

    std::vector<int> physical_dims;
    std::vector<int> queues_per_dim;

    // collective communication
    CollectiveImplLookup* collective_impl_lookup;
    int num_streams;
    static uint8_t* dummy_data;
    std::map<std::string, LogicalTopology*> logical_topologies;
    CollectiveOptimization collectiveOptimization;
    Tick last_scheduled_collective;

    // statistics
    bool trace_enabled;

    // （2026-10-01）无消费方：整节点 replay 分派随 Path-2 清除删除；键
    // 仅为模板兼容保留在 known_system_keys 中并被解析，任何取值都不再
    // 驱动行为。
    bool replay_only;
};

}  // namespace AstraSim

#endif /* __SYSTEM_HH__ */
