/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __SYSTEM_HH__
#define __SYSTEM_HH__

#include <chrono>

#include "astra-sim/common/AstraNetworkAPI.hh"
#include "astra-sim/common/AstraRemoteMemoryAPI.hh"
#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/CollectivePhase.hh"
#include "astra-sim/system/CommunicatorGroup.hh"
#include "astra-sim/system/MemBus.hh"
#include "astra-sim/system/Roofline.hh"
#include "astra-sim/system/UsageTracker.hh"
#include "astra-sim/system/astraccl/CollectiveImplLookup.hh"
#include "astra-sim/system/astraccl/native_collectives/logical_topology/RingTopology.hh"
#include "astra-sim/workload/Workload.hh"

namespace AstraSim {

class BaseStream;
class StreamBaseline;
class DataSet;
class QueueLevels;
class Workload;
class LogicalTopology;
class BasicLogicalTopology;
class OfflineGreedy;

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
        void notify_stream_removed(int vnet, Tick running_time);

        Sys* sys;
        int max_running_streams;
        int ready_list_threshold;
        int queue_threshold;
        std::map<int, int> running_streams;
        std::map<int, std::list<BaseStream*>::iterator> stream_pointer;
        std::vector<uint64_t> total_active_chunks_per_dimension;
        std::map<int, int> queue_id_to_dimension;
        std::vector<UsageTracker> usage;
    };
    //---------------------------------------------------------------------------

    // Constructor / Destructor
    // -------------------------------------------------
    // execution_mode/graph_source: step-1-2 execution-mode factory
    // (ExecutionMode.hh / GraphSource.hh). The Static default keeps the
    // legacy main.cc call site and the byte-for-byte static behavior
    // unchanged; Online mode never constructs the ETFeeder and never
    // requires .et files.
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
            ExecutionDriven::ExecutionMode::Static,
        std::shared_ptr<ExecutionDriven::GraphSource> graph_source = nullptr);
    ~Sys();
    //---------------------------------------------------------------------------

    // Intialization
    // ------------------------------------------------------------
    bool initialize_sys(std::string name);
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
                                              InjectionPolicy injection_policy,
                                              CollectiveImpl* collective_impl,
                                              CommunicatorGroup* comm_group = nullptr);
    //---------------------------------------------------------------------------

    // Middle-level Network Primitives
    // ------------------------------------------
    uint64_t determine_chunk_size(uint64_t& size, ComType type);
    int get_priority(int explicit_priority);
    void insert_into_ready_list(BaseStream* stream);
    void insert_stream(std::list<BaseStream*>* queue, BaseStream* baseStream);
    void ask_for_schedule(int max);
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
    bool initialized;

    // workload
    Workload* workload;

    // step-1-2 execution-mode factory state (see constructor comment).
    ExecutionDriven::ExecutionMode execution_mode_ =
        ExecutionDriven::ExecutionMode::Static;
    std::shared_ptr<ExecutionDriven::GraphSource> graph_source_ = nullptr;
    // step-1-8 replay-clock scope (main ruling 2026-08-15): true only for
    // --online-mode replay; strategy mode keeps real physics (false).
    // roofline model
    bool roofline_enabled;
    double peak_perf;
    Roofline* roofline;

    // memory
    bool track_local_mem;
    std::string local_mem_trace_filename;
    double local_mem_bw;
    uint64_t local_mem_latency;
    // Multi-user local-HBM bandwidth contention (system config key
    // "hbm-bandwidth-contention", default true). When true, COMP roofline
    // traffic and NoC p2p comm endpoint reads/writes compete for the single
    // local-mem-bw scalar through LocalHbmBandwidthModel (strict fair
    // sharing, event-driven re-allocation). local-mem-bw <= 0 force-disables
    // it (a zero-rate fluid model would stall forever); false restores the
    // legacy closed-form roofline + comm-without-HBM behavior.
    bool hbm_bandwidth_contention;
    double remote_mem_bw;
    uint64_t remote_mem_latency;
    double pipeline_tile_fraction;
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
    double comm_scale;
    bool rendezvous_enabled;

    // scheduler
    SchedulerUnit* scheduler_unit;
    QueueLevels* vLevels;
    OfflineGreedy* offline_greedy;
    IntraDimensionScheduling intra_dimension_scheduling;
    InterDimensionScheduling inter_dimension_scheduling;
    int round_robin_inter_dimension_scheduler;
    int active_chunks_per_dimension;
    int priority_counter;
    uint64_t pending_events;
    int preferred_dataset_splits;
    int concurrent_streams;
    int active_first_phase;
    int max_running;

    // for supporting LIFO
    std::list<BaseStream*> ready_list;
    SchedulingPolicy scheduling_policy = SchedulingPolicy::FIFO;
    int first_phase_streams;
    int total_running_streams;
    std::map<int, std::list<BaseStream*>> active_Streams;
    std::map<int, std::list<int>> stream_priorities;

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
    int dim_to_break;
    std::vector<int> logical_broken_dims;

    std::vector<int> physical_dims;
    std::vector<int> queues_per_dim;

    // collective communication
    CollectiveImplLookup* collective_impl_lookup;
    int num_streams;
    static uint8_t* dummy_data;
    std::map<std::string, LogicalTopology*> logical_topologies;
    CollectiveOptimization collectiveOptimization =
        CollectiveOptimization::Baseline;
    Tick last_scheduled_collective;

    // statistics
    bool trace_enabled;

    // skip simulation for all nodes and use current duration
    bool replay_only;
};

}  // namespace AstraSim

#endif /* __SYSTEM_HH__ */
