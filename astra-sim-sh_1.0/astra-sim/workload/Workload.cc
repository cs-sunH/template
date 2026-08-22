/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/workload/Workload.hh"

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/IntData.hh"
#include "astra-sim/system/MemEventHandlerData.hh"
#include "astra-sim/system/RecvPacketEventHandlerData.hh"
#include "astra-sim/system/SendPacketEventHandlerData.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include "astra-sim/workload/MetricCollector.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include <json/json.hpp>

#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>

using namespace std;
using namespace AstraSim;
using namespace Chakra::FeederV3;
using json = nlohmann::json;

typedef ChakraProtoMsg::NodeType ChakraNodeType;
typedef ChakraProtoMsg::CollectiveCommType ChakraCollectiveCommType;

namespace {

// Step 1-5: node-terminal record with the online-mode reverse index.
// Online mode reads request_id/stage/generation from the GraphSource view
// (NodeStore-backed); static mode stays nullptr/0 -- byte-for-byte the
// pre-phase-1 shape (the lookup is gated to online mode, so the static path
// performs no extra attribute reads). The view (if any) is kept alive for
// the whole call so the c_str() pointers are valid for the duration of the
// record (the hook runs inside record_node_terminal).
void record_node_terminal(
    std::shared_ptr<ExecutionDriven::GraphSource> graph_source,
    ExecutionDriven::ExecutionMode mode, int rank, uint64_t node_id,
    ExecutionDriven::NodeTerminalStatus status) {
    const ExecutionDriven::NodeView* nv = nullptr;
    if (mode == ExecutionDriven::ExecutionMode::Online) {
        nv = graph_source->lookup_ptr(node_id);
    }
    ExecutionDriven::CompletionObserver::instance().record_node_terminal(
        rank, node_id,
        (nv != nullptr && !nv->request_id.empty()) ? nv->request_id.c_str()
                                                    : nullptr,
        (nv != nullptr && !nv->stage.empty()) ? nv->stage.c_str() : nullptr,
        nv != nullptr ? nv->generation : 0, Sys::boostedTick(), status);
}

}  // namespace

Workload::Workload(Sys* sys, string et_filename, string comm_group_filename,
                   ExecutionDriven::ExecutionMode execution_mode,
                   std::shared_ptr<ExecutionDriven::GraphSource> graph_source) {
    this->workload_logger_ = LoggerFactory::get_logger("workload");
    this->execution_mode_ = execution_mode;
    this->graph_source_ = std::move(graph_source);

    string workload_filename = et_filename + "." + to_string(sys->id) + ".et";
    if (execution_mode == ExecutionDriven::ExecutionMode::Online) {
        // Online mode (step-1-2 execution-mode factory): no ETFeeder, no .et
        // file requirement; the dynamic GraphSource was injected at Sys
        // creation (NodeStore-backed implementation lands in step 1-4).
        this->et_feeder = nullptr;
    } else {
        // Static path, byte-for-byte the pre-phase-1 behavior.
        // Check if workload filename exists
        if (access(workload_filename.c_str(), R_OK) < 0) {
            string error_msg;
            if (errno == ENOENT) {
                error_msg =
                    "workload file: " + workload_filename + " does not exist";
            } else if (errno == EACCES) {
                error_msg = "workload file: " + workload_filename +
                            " exists but is not readable";
            } else {
                error_msg =
                    "Unknown workload file: " + workload_filename +
                    " access error";
            }
            workload_logger_->critical(error_msg);
            exit(EXIT_FAILURE);
        }
        this->et_feeder = new ETFeeder(workload_filename);
    }
    // Step 1-4: the GraphSource is the sole dependency-state owner. Static
    // mode wraps the ETFeeder + DependancyResolver in the adapter
    // (ETFeederGraphSource) -- byte-for-byte the pre-phase-1 resolver calls;
    // online mode received an injected NodeStore-backed source at Sys
    // creation and must never fall back to an ETFeeder.
    if (this->graph_source_ == nullptr &&
        execution_mode == ExecutionDriven::ExecutionMode::Static) {
        this->graph_source_ =
            std::make_shared<ExecutionDriven::ETFeederGraphSource>(
                this->et_feeder, sys->id);
    }
    this->comm_groups.clear();
    // TODO: parametrize the number of available hardware resources
    this->hw_resource = new HardwareResource(1, sys->id);
    this->local_mem_usage_tracker =
        std::make_unique<LocalMemUsageTracker>(sys->id);
    this->sys = sys;
    if (sys->hbm_bandwidth_contention) {
        // Multi-user local-HBM contention is an execution model assembled
        // in BOTH modes (static and online), identical timing/params; the
        // flag is auto-false for local-mem-bw <= 0 (Sys::initialize_sys).
        this->local_hbm_bandwidth_model =
            std::make_unique<LocalHbmBandwidthModel>(sys, this);
    }
    // Step 1-8: the local_mem tracker is ETFeederNode-bound (no online
    // NodeView overloads); online mode with track_local_mem fails closed
    // instead of dereferencing null handles at the record sites. Placed
    // after this->sys = sys (the tracker is constructed above regardless;
    // recordStart/recordEnd never run in online mode).
    if (this->sys->track_local_mem &&
        execution_mode == ExecutionDriven::ExecutionMode::Online) {
        workload_logger_
            ->critical("track_local_mem is not supported in online mode "
                       "(step 1-8; the local_mem tracker is ETFeederNode-"
                       "bound)");
        exit(EXIT_FAILURE);
    }
    initialize_comm_groups(comm_group_filename);
    this->stats = new Statistics(this);
    this->is_finished = false;
}

Workload::~Workload() {
    for (auto comm_group : comm_groups) {
        delete comm_group.second;
    }
    comm_groups.clear();

    if (this->et_feeder != nullptr) {
        delete this->et_feeder;
    }
    if (this->hw_resource != nullptr) {
        delete this->hw_resource;
    }
    if (this->stats != nullptr) {
        delete this->stats;
    }
}

void Workload::initialize_comm_groups(string comm_group_filename) {
    // communicator group input file is not given
    if (comm_group_filename.find("empty") != std::string::npos) {
        comm_groups.clear();
        return;
    }

    ifstream inFile;
    json j;
    inFile.open(comm_group_filename);
    inFile >> j;

    for (json::iterator it = j.begin(); it != j.end(); ++it) {
        std::string comm_group_name = it.key();
        int comm_group_id = std::stoi(comm_group_name);

        std::vector<int> involved_NPUs;
        std::vector<int> dimension_sizes;
        if (it.value().is_array()) {
            involved_NPUs = it.value().get<std::vector<int>>();
        } else if (it.value().is_object()) {
            involved_NPUs =
                it.value().at("ranks").get<std::vector<int>>();
            if (it.value().contains("dimensions")) {
                dimension_sizes = it.value()
                                      .at("dimensions")
                                      .get<std::vector<int>>();
            }
        } else {
            throw std::runtime_error(
                "Communicator group must be a rank array or an object with "
                "ranks and dimensions");
        }

        comm_groups[comm_group_id] = new CommunicatorGroup(
            comm_group_id, involved_NPUs, sys, dimension_sizes);
    }
}

void Workload::issue_pytorch_pg_metadata(
    const ExecutionDriven::NodeView& node) {
    // For read comm groups from torch, might overwrite previous.
    std::string pg_info = node.inputs_values;
    if (pg_info.empty()) {
        return;
    }
    pg_info = pg_info.substr(2, pg_info.size() - 4);

    try {
        json valuesRoot = json::parse(pg_info);

        for (const auto& item : valuesRoot) {
            std::string pgName = item.at("pg_name").get<std::string>();
            std::vector<int> involved_NPUs =
                item.at("ranks").get<std::vector<int>>();

            if (involved_NPUs.empty()) {
                for (int i = 0; i < sys->total_nodes; i++) {
                    involved_NPUs.push_back(i);
                }
            }

            int32_t pgNameInt = std::stoi(pgName);
            // To ensure pgName > 0
            CommunicatorGroup* cg =
                new CommunicatorGroup(pgNameInt + 1, involved_NPUs, sys);
            this->comm_groups[pgNameInt] = cg;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing or processing JSON: " << e.what()
                  << std::endl;
    }
}

void Workload::issue_dep_free_nodes() {
    // Step 1-4: iterate the GraphSource free views (ascending id, byte-exact
    // order of the pre-phase-1 std::set copy). Static auto-advance only --
    // call() gates this behind the execution mode; online mode issues only
    // through the post-commit deferred path (steps 1-6/1-11), which calls
    // this function directly per rank.
    graph_source_->for_each_dep_free([&](const auto& nv) {
        if (execution_mode_ == ExecutionDriven::ExecutionMode::Online) {
            // Step 1-8: online availability check on the NodeView
            // (et_node is nullptr in online mode). Strategy mode keeps the
            // real serialized physics (path-2 removal 2026-08-18 deleted the
            // replay-only concurrent calibrated-COMP bypass).
            if (hw_resource->is_available(nv)) {
                issue(nv);
            }
        } else {
            auto node = graph_source_->et_node(nv.global_id);
            if (hw_resource->is_available(node)) {
                issue(nv);
            }
        }
    });
}

void Workload::issue(const ExecutionDriven::NodeView& node) {
    auto logger = workload_logger_;
    if (sys->trace_enabled) {
        logger->debug("issue,sys->id={}, tick={}, node->id={}, "
                      "node->name={}, node->type={}",
                      sys->id, Sys::boostedTick(), node.global_id, node.name,
                      static_cast<uint64_t>(node.node_type));
    }

    graph_source_->take_node(node.global_id);
    // Static path: the ETFeederNode handle for the node-bound consumers
    // (HardwareResource / Statistics / local_mem tracker) is fetched through
    // the GraphSource. Step 1-8: the online path (et_node == nullptr) uses
    // the NodeView overloads.
    auto et_node = graph_source_->et_node(node.global_id);
    if (execution_mode_ == ExecutionDriven::ExecutionMode::Online) {
        this->hw_resource->occupy(node);
        // stats->record_end will be called in Workload::call
        stats->record_start(node, Sys::boostedTick());
    } else {
        this->hw_resource->occupy(et_node);
        // stats->record_end will be called in Workload::call
        stats->record_start(et_node, Sys::boostedTick());
    }
    // Side-band metrics observation only; does not touch the node, the
    // dependency resolver, or the event queue (doc sec.5.5).
    if (MetricCollector::instance().enabled()) {
        MetricCollector::instance().on_node_issue(sys->id, node.global_id,
                                                  Sys::boostedTick());
    }
    if (this->sys->track_local_mem) {
        this->local_mem_usage_tracker->recordStart(et_node,
                                                   Sys::boostedTick());
    }
    if (sys->replay_only) {
        issue_replay(node);
    } else {
        if ((node.node_type == ChakraNodeType::MEM_LOAD_NODE) ||
            (node.node_type == ChakraNodeType::MEM_STORE_NODE)) {
            issue_remote_mem(node);
        } else if (node.node_type == ChakraNodeType::COMP_NODE) {
            if (!this->sys->roofline_enabled) {
                issue_replay(node);
            } else {
                if (node.is_cpu_op) {
                    // comp node on cpu
                    // should only appears in real system trace and should run
                    // with replay.
                    issue_replay(node);
                } else {
                    // comp node on gpu
                    issue_comp(node);
                }
            }
        } else if (node.node_type == ChakraNodeType::COMM_COLL_NODE ||
                   node.node_type == ChakraNodeType::COMM_SEND_NODE ||
                   node.node_type == ChakraNodeType::COMM_RECV_NODE) {
            issue_comm(node);
        } else if (node.node_type == ChakraNodeType::INVALID_NODE) {
            skip_invalid(node);
        } else if (node.node_type == ChakraNodeType::METADATA_NODE) {
            issue_metadata(node);
        } else {
            logger->critical("Unknown node type");
            exit(EXIT_FAILURE);
        }
    }
}

void Workload::issue_metadata(const ExecutionDriven::NodeView& node) {
    // TODO: someway to identify this metadata node is a pytorch pg node
    if (true) {
        issue_pytorch_pg_metadata(node);
    } else {
        throw std::runtime_error("Unknown metadata node type");
    }
    this->skip_invalid(node);  // for proper dependancy resolving
}

void Workload::issue_replay(const ExecutionDriven::NodeView& node) {
    WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
    wlhd->node_id = node.global_id;
    uint64_t runtime = 1ul;
    if (node.compute.runtime_ns != 0ul) {
        // chakra runtimes are in microseconds and the GraphSource adapter
        // already converted them into nanoseconds
        runtime = node.compute.runtime_ns;
    }
    if (node.is_cpu_op) {
        hw_resource->tics_cpu_ops += runtime;
    } else {
        hw_resource->tics_gpu_ops += runtime;
    }
    sys->register_event(this, EventType::General, wlhd, runtime);
}

void Workload::issue_remote_mem(const ExecutionDriven::NodeView& node) {
    WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
    wlhd->sys_id = sys->id;
    wlhd->workload = this;
    wlhd->node_id = node.global_id;
    // Path-2 removal (2026-08-18): the replay-only instant (1ns) remote
    // MEM completion branch was deleted with the replay route; strategy and
    // the static path keep the real 26-port remote-memory FIFO physics.
    //
    // Local-HBM contention: the ET int attr "hbm-access-mode" marks the
    // edge rank that is the LOCAL data endpoint of a remote-pool transfer
    // (1 = read job, 2 = write job, bytes = tensor_size; 0/absent = the
    // rank only passes data through and is not charged). Node completion
    // becomes the join of the remote-port transaction and this HBM job.
    // Armed before remote_mem->issue so a synchronous completion cannot
    // race the join record.
    if (local_hbm_bandwidth_model != nullptr &&
        node.compute.hbm_access_mode != 0 &&
        node.compute.tensor_size > 0) {
        arm_hbm_join(node.global_id);
        WorkloadLayerHandlerData* job_wlhd =
            make_hbm_job_wlhd(node.global_id);
        if (node.compute.hbm_access_mode == 1) {
            local_hbm_bandwidth_model->issue_pool_read(
                node.compute.tensor_size, job_wlhd);
        } else {
            local_hbm_bandwidth_model->issue_pool_write(
                node.compute.tensor_size, job_wlhd);
        }
    }
    //
    // Phase-4 sensing (方案 §6.2 操作 2): remote-FIFO 实账本层 issue record.
    // Pure observation (counters only). Static mode's
    // MEM traffic also flows through here and is accounted the same way --
    // the layer mirrors AnalyticalRemoteMemory's own port state exactly
    // (see Workload.hh for the single-server reconstruction argument).
    if (execution_mode_ == ExecutionDriven::ExecutionMode::Online) {
        ExecutionDriven::RemoteFifoLedger::instance().record_issue(
            sys->id, node.compute.tensor_size);
    }
    sys->remote_mem->issue(node.compute.tensor_size, wlhd);
}

namespace AstraSim {
namespace ExecutionDriven {

RemoteFifoLedger& RemoteFifoLedger::instance() {
    static RemoteFifoLedger ledger;
    return ledger;
}

void RemoteFifoLedger::set_enabled(const bool enabled) {
    enabled_ = enabled;
}

void RemoteFifoLedger::record_issue(const int sys_id,
                                    const uint64_t tensor_size) {
    if (!enabled_) {
        return;
    }
    PortCounters& port = ports_[sys_id];
    port.issued_count += 1;
    port.issued_bytes += tensor_size;
    const uint64_t in_flight =
        port.issued_count - port.completed_count;
    const uint64_t in_flight_bytes =
        port.issued_bytes - port.completed_bytes;
    if (in_flight > port.peak_in_flight_count) {
        port.peak_in_flight_count = in_flight;
    }
    if (in_flight_bytes > port.peak_in_flight_bytes) {
        port.peak_in_flight_bytes = in_flight_bytes;
    }
}

void RemoteFifoLedger::record_completion(const int sys_id,
                                         const uint64_t tensor_size) {
    if (!enabled_) {
        return;
    }
    PortCounters& port = ports_[sys_id];
    port.completed_count += 1;
    port.completed_bytes += tensor_size;
}

std::vector<int> RemoteFifoLedger::active_ports() const {
    std::vector<int> result;
    result.reserve(ports_.size());
    for (const auto& [sys_id, counters] : ports_) {
        (void)counters;
        result.push_back(sys_id);
    }
    return result;  // std::map iterates in key order -- deterministic
}

const RemoteFifoLedger::PortCounters* RemoteFifoLedger::port(
    const int sys_id) const {
    const auto it = ports_.find(sys_id);
    return it == ports_.end() ? nullptr : &it->second;
}

uint64_t RemoteFifoLedger::total_issued_count() const {
    uint64_t total = 0;
    for (const auto& [sys_id, counters] : ports_) {
        (void)sys_id;
        total += counters.issued_count;
    }
    return total;
}

uint64_t RemoteFifoLedger::total_issued_bytes() const {
    uint64_t total = 0;
    for (const auto& [sys_id, counters] : ports_) {
        (void)sys_id;
        total += counters.issued_bytes;
    }
    return total;
}

uint64_t RemoteFifoLedger::total_completed_count() const {
    uint64_t total = 0;
    for (const auto& [sys_id, counters] : ports_) {
        (void)sys_id;
        total += counters.completed_count;
    }
    return total;
}

uint64_t RemoteFifoLedger::total_completed_bytes() const {
    uint64_t total = 0;
    for (const auto& [sys_id, counters] : ports_) {
        (void)sys_id;
        total += counters.completed_bytes;
    }
    return total;
}

bool RemoteFifoLedger::drained() const {
    for (const auto& [sys_id, counters] : ports_) {
        (void)sys_id;
        if (counters.issued_count != counters.completed_count) {
            return false;
        }
    }
    return true;
}

}  // namespace ExecutionDriven
}  // namespace AstraSim

void Workload::issue_comp(const ExecutionDriven::NodeView& node) {
    if (!this->sys->roofline_enabled) {
        throw std::runtime_error(
            "Roofline model is not enabled for non-replay comp");
    }

    if (node.is_cpu_op) {
        throw std::runtime_error("Roofline is only available for GPU nodes");
        return;
    }

    WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
    wlhd->sys_id = sys->id;
    wlhd->workload = this;
    wlhd->node_id = node.global_id;

    const uint64_t node_num_ops = node.compute.num_ops;
    const uint64_t node_tensor_size = node.compute.tensor_size;
    double num_ops = static_cast<double>(node_num_ops);
    double tensor_size = static_cast<double>(node_tensor_size);
    // if tensor_size is 0 during roofline mode, this is an invalid node
    if (tensor_size == 0) {
        skip_invalid(node);
        return;
    }

    // Side-band accumulation of values already read above; only nodes that
    // really execute as GPU compute are counted (doc sec.5.5).
    if (MetricCollector::instance().enabled()) {
        MetricCollector::instance().on_compute_issue(sys->id, node_num_ops,
                                                     node_tensor_size);
    }

    double operational_intensity = num_ops / tensor_size;
    double perf = sys->roofline->get_perf(operational_intensity);
    double compute_elapsed_time = num_ops / perf;  // sec
    if (sys->local_mem_latency > 0) {
        const double compute_only_elapsed_time = num_ops / sys->peak_perf;
        const double local_mem_elapsed_time =
            (static_cast<double>(sys->local_mem_latency) / 1e9) +
            (tensor_size / sys->local_mem_bw);
        compute_elapsed_time =
            std::max(compute_only_elapsed_time, local_mem_elapsed_time);
    }
    double elapsed_time = compute_elapsed_time;

    if (node.compute.has_remote_weight_bytes) {
        if (local_hbm_bandwidth_model != nullptr) {
            // The remote-operand pipeline roofline and the shared-HBM fluid
            // model are two different timing authorities for the same COMP
            // node; combining them is a configuration error, not a data
            // property.  (This repository materializes traces with
            // remote_operand_loads=false, so the branch is unreachable in
            // the standard pipeline -- kept fail-closed for parity.)
            throw std::runtime_error(
                "HBM bandwidth contention cannot be combined with remote "
                "operand pipeline loads on the same COMP node");
        }
        if (sys->remote_mem_bw <= 0) {
            throw std::runtime_error(
                "Pipeline roofline requires remote-mem-bw in system config");
        }

        double remote_weight_bytes =
            static_cast<double>(node.compute.remote_weight_bytes);
        double first_tile_time =
            (static_cast<double>(sys->remote_mem_latency) / 1e9) +
            (sys->pipeline_tile_fraction * remote_weight_bytes) /
                sys->remote_mem_bw;
        double remaining_transfer_time =
            ((1.0 - sys->pipeline_tile_fraction) * remote_weight_bytes) /
            sys->remote_mem_bw;
        elapsed_time = first_tile_time +
            std::max(remaining_transfer_time, compute_elapsed_time);
    }

    uint64_t runtime = static_cast<uint64_t>(elapsed_time * 1e9);  // sec -> ns
    // Step 1-8: online execution-driven calibration. The online GraphBatch
    // (graph_batch_builder.py) carries planner-LUT-aligned durations so the
    // engine timeline is order-isomorphic to the Python decision sequence
    // (fail-closed order consumption). The static .et
    // path never sets runtime_ns (duration_micros=0, parsed as 0) and keeps
    // the roofline above -- byte-for-byte preserved. Same honor pattern as
    // issue_replay's runtime_ns branch.
    if (node.compute.runtime_ns != 0ul) {
        runtime = node.compute.runtime_ns;
    }
    // Multi-user local-HBM contention: an uncalibrated roofline COMP drains
    // through the per-rank fluid model (bytes compete with every other HBM
    // job of this rank; ops drain at peak_perf in parallel, preserving the
    // max() semantics for a single user). COMP nodes carrying a non-zero
    // calibrated runtime_ns keep the calibrated duration instead (the
    // engine timeline stays order-isomorphic to the planner decision log;
    // this repo's traces have no calibrated COMP, so the fluid path is the
    // live one).
    if (local_hbm_bandwidth_model != nullptr &&
        node.compute.runtime_ns == 0ul) {
        // tics_gpu_ops is accumulated by the model at completion (elapsed
        // wall time incl. sharing waits), replacing the issue-time tics.
        local_hbm_bandwidth_model->issue_compute(
            node_num_ops, node_tensor_size, wlhd);
    } else {
        if (node.is_cpu_op) {
            hw_resource->tics_cpu_ops += runtime;
        } else {
            hw_resource->tics_gpu_ops += runtime;
        }
        sys->register_event(this, EventType::General, wlhd, runtime);
    }

    auto& op_stat = this->stats->get_operator_statistics(node.global_id);
    op_stat.operation_intensity = operational_intensity;
    op_stat.compute_utilization = perf / sys->peak_perf;
    op_stat.memory_utilization =
        (perf / operational_intensity) / sys->local_mem_bw;
    op_stat.is_memory_bound = perf < sys->peak_perf;
    workload_logger_
        ->debug("operation_intensity={}, perf={}, elapsed_time={} "
                "local_mem_latency_ns={} "
                "compute_utilization={} memory_utilization={} tensor_size={} "
                "num_ops={}",
                operational_intensity, perf, elapsed_time,
                sys->local_mem_latency,
                op_stat.compute_utilization.value(),
                op_stat.memory_utilization.value(), tensor_size, num_ops);
}

void Workload::issue_comm(const ExecutionDriven::NodeView& node) {
    if (node.is_cpu_op) {
        throw std::runtime_error("Comm node should not be on CPU");
    }
    // Path-2 removal (2026-08-18): the replay-only instant (1ns) comm
    // completion branch was deleted with the replay route; strategy mode
    // keeps the real network physics.
    const auto node_type = node.node_type;
    if (node_type == ChakraNodeType::COMM_COLL_NODE) {
        this->issue_coll_comm(node);
    } else if (node_type == ChakraNodeType::COMM_SEND_NODE) {
        this->issue_send_comm(node);
    } else if (node_type == ChakraNodeType::COMM_RECV_NODE) {
        this->issue_recv_comm(node);
    } else {
        throw std::runtime_error("Unknown comm node type");
    }
}

void Workload::issue_coll_comm(const ExecutionDriven::NodeView& node) {
    // involved_dim was parsed by the GraphSource adapter (step 1-4) with the
    // legacy semantics: bool_list type check (cerr + exit on mismatch) or the
    // 4x-true default when the attribute is absent.
    const std::vector<bool>& involved_dims = node.coll.involved_dim;

    CommunicatorGroup* comm_group = extract_comm_group(node);
    const auto comm_type =
        static_cast<ChakraCollectiveCommType>(node.coll.comm_type);
    const auto comm_size = node.coll.bytes;
    // Record communication size for bandwidth calculation
    stats->get_operator_statistics(node.global_id).comm_size = comm_size;
    // TODO: comm_tag? which is used to distinguish two different collective in
    // same pg
    const auto comm_priority = node.coll.priority;  // default 0u

    if (comm_type == ChakraCollectiveCommType::ALL_REDUCE) {
        DataSet* fp = sys->generate_all_reduce(comm_size, involved_dims,
                                               comm_group, comm_priority, node.global_id);
        collective_comm_node_id_map[fp->my_id] = node.global_id;
        collective_comm_wrapper_map[fp->my_id] = fp;
        fp->set_notifier(this, EventType::CollectiveCommunicationFinished);
    } else if (comm_type == ChakraCollectiveCommType::ALL_TO_ALL) {
        DataSet* fp = sys->generate_all_to_all(comm_size, involved_dims,
                                               comm_group, comm_priority, node.global_id);
        collective_comm_node_id_map[fp->my_id] = node.global_id;
        collective_comm_wrapper_map[fp->my_id] = fp;
        fp->set_notifier(this, EventType::CollectiveCommunicationFinished);
    } else if (comm_type == ChakraCollectiveCommType::ALL_GATHER) {
        DataSet* fp = sys->generate_all_gather(comm_size, involved_dims,
                                               comm_group, comm_priority, node.global_id);
        collective_comm_node_id_map[fp->my_id] = node.global_id;
        collective_comm_wrapper_map[fp->my_id] = fp;
        fp->set_notifier(this, EventType::CollectiveCommunicationFinished);
    } else if (comm_type == ChakraCollectiveCommType::REDUCE_SCATTER) {
        DataSet* fp = sys->generate_reduce_scatter(comm_size, involved_dims,
                                                   comm_group, comm_priority, node.global_id);
        collective_comm_node_id_map[fp->my_id] = node.global_id;
        collective_comm_wrapper_map[fp->my_id] = fp;
        fp->set_notifier(this, EventType::CollectiveCommunicationFinished);
    } else if (comm_type == ChakraCollectiveCommType::BROADCAST) {
        // TODO: implement broadcast, for now just replay
        uint64_t runtime = 1ul;
        if (node.compute.runtime_ns != 0ul) {
            // chakra runtimes are in microseconds and the GraphSource adapter
            // already converted them into nanoseconds
            runtime = node.compute.runtime_ns;
        }
        DataSet* fp = new DataSet(1);
        fp->set_notifier(this, EventType::CollectiveCommunicationFinished);
        collective_comm_node_id_map[fp->my_id] = node.global_id;
        collective_comm_wrapper_map[fp->my_id] = fp;
        sys->register_event(fp, EventType::General, nullptr,
                            // chakra runtimes are in microseconds and we
                            // should convert it into nanoseconds
                            runtime);
        fp->set_notifier(this, EventType::CollectiveCommunicationFinished);
    } else {
        throw std::runtime_error("Unsupported collective comm type");
    }
}

void Workload::issue_send_comm(const ExecutionDriven::NodeView& node) {
    const auto src = node.comm.src;  // adapter default: this rank
    if (src != this->sys->id) {
        throw std::runtime_error("Send node should be issued by the sender");
    }
    const auto dst = node.comm.dst;
    const auto size = node.comm.bytes;
    // Record communication size for bandwidth calculation
    stats->get_operator_statistics(node.global_id).comm_size = size;
    const auto tag = node.comm.tag;

    // Local-HBM contention: the SENDER is a data endpoint of the p2p
    // transfer -- its HBM read competes with every other HBM job of this
    // rank (bytes = comm bytes; a 1B ACK is charged like any other byte
    // stream). bytes == 0 creates no job; the ET bool attr "hbm-charge"
    // (default true) opts the endpoint out (NoC<->SerDes pass-through on
    // edge ranks, whose stream is charged at its real data endpoint).
    // Node completion becomes the join of the network-side PacketSent and
    // this HBM job. Armed before sim_send so a synchronous completion
    // cannot race the join record.
    if (local_hbm_bandwidth_model != nullptr && size > 0 &&
        node.comm.hbm_charge) {
        arm_hbm_join(node.global_id);
        local_hbm_bandwidth_model->issue_comm_read(
            size, make_hbm_job_wlhd(node.global_id));
    }

    sim_request snd_req;
    snd_req.srcRank = src;
    snd_req.dstRank = dst;
    snd_req.reqType = UINT8;
    SendPacketEventHandlerData* sehd = new SendPacketEventHandlerData;
    sehd->callable = this;
    sehd->wlhd = new WorkloadLayerHandlerData;
    sehd->wlhd->node_id = node.global_id;
    sehd->event = EventType::PacketSent;
    sys->front_end_sim_send(0, Sys::dummy_data, size, UINT8, dst, tag, &snd_req,
                            Sys::FrontEndSendRecvType::NATIVE,
                            &Sys::handleEvent, sehd);
}

void Workload::issue_recv_comm(const ExecutionDriven::NodeView& node) {
    const auto src = node.comm.src;
    const auto dst = node.comm.dst;  // adapter default: this rank
    if (dst != this->sys->id) {
        throw std::runtime_error("Recv node should be issued by the receiver");
    }
    const auto size = node.comm.bytes;
    // Record communication size for bandwidth calculation
    stats->get_operator_statistics(node.global_id).comm_size = size;
    const auto tag = node.comm.tag;

    // Local-HBM contention: the RECEIVER is a data endpoint -- its HBM write
    // competes with every other HBM job of this rank. Same rules as the
    // send side (hbm-charge default true, bytes == 0 -> no job).
    if (local_hbm_bandwidth_model != nullptr && size > 0 &&
        node.comm.hbm_charge) {
        arm_hbm_join(node.global_id);
        local_hbm_bandwidth_model->issue_comm_write(
            size, make_hbm_job_wlhd(node.global_id));
    }

    sim_request rcv_req;
    RecvPacketEventHandlerData* rcehd = new RecvPacketEventHandlerData;
    rcehd->wlhd = new WorkloadLayerHandlerData;
    rcehd->wlhd->node_id = node.global_id;
    rcehd->workload = this;
    rcehd->event = EventType::PacketReceived;
    sys->front_end_sim_recv(0, Sys::dummy_data, size, UINT8, src, tag, &rcv_req,
                            Sys::FrontEndSendRecvType::NATIVE,
                            &Sys::handleEvent, rcehd);
}

void Workload::skip_invalid(const ExecutionDriven::NodeView& node) {
    const auto node_id = node.global_id;
    // Step 1-3: unconditional node-terminal record (独立于 metrics 开关).
    // Step 1-5: the online path fills the reverse index from the GraphSource
    // view; static stays nullptr/0. Skipped is explicit: the node never
    // executed (INVALID_NODE / metadata / zero tensor_size); whether Skipped
    // satisfies a watch is the watch's own policy, never a default.
    record_node_terminal(graph_source_, execution_mode_, sys->id, node_id,
                         ExecutionDriven::NodeTerminalStatus::Skipped);
    // Step 1-4: the GraphSource is the sole dependency-state owner.
    graph_source_->finish_node(node_id);
    auto logger = workload_logger_;
    logger->debug("callback,sys->id={}, tick={}, node->id={}, "
                  "node->name={}, node->type={}",
                  sys->id, Sys::boostedTick(), node.global_id, node.name,
                  static_cast<uint64_t>(node.node_type));
    // Step 1-8: online path (et_node == nullptr) releases through the
    // NodeView overloads. The track_local_mem block below keeps the static
    // et_node handle (unreachable in online mode: the ctor fails closed on
    // track_local_mem + Online).
    std::shared_ptr<Chakra::FeederV3::ETFeederNode> et_node = nullptr;
    if (execution_mode_ == ExecutionDriven::ExecutionMode::Online) {
        const ExecutionDriven::NodeView* nv =
            graph_source_->lookup_ptr(node_id);
        if (nv == nullptr) {
            workload_logger_
                ->critical("skip_invalid for unknown online node id={}",
                           node_id);
            exit(EXIT_FAILURE);
        }
        hw_resource->release(*nv);
        stats->record_end(*nv, Sys::boostedTick());
    } else {
        et_node = graph_source_->et_node(node_id);
        hw_resource->release(et_node);
        stats->record_end(et_node, Sys::boostedTick());
    }
    if (MetricCollector::instance().enabled()) {
        MetricCollector::instance().on_node_complete(sys->id, node_id,
                                                     Sys::boostedTick());
    }
    if (this->sys->track_local_mem) {
        this->local_mem_usage_tracker->recordEnd(et_node, Sys::boostedTick());
    }
}

void Workload::call(EventType event, CallData* data) {
    if (is_finished) {
        return;
    }

    if (event == EventType::CollectiveCommunicationFinished) {
        IntData* int_data = (IntData*)data;
        uint64_t coll_comm_id = int_data->data;

        hw_resource->tics_gpu_comms += int_data->execution_time;
        uint64_t node_id = collective_comm_node_id_map[coll_comm_id];
        // Step 1-8: online mode has no ETFeederNode handle (et_node ==
        // nullptr); the online branch releases / records through the
        // NodeView. The static branch below stays byte-identical.
        const ExecutionDriven::NodeView* nv = nullptr;
        shared_ptr<Chakra::FeederV3::ETFeederNode> node = nullptr;
        if (execution_mode_ == ExecutionDriven::ExecutionMode::Online) {
            nv = graph_source_->lookup_ptr(node_id);
            if (nv == nullptr) {
                workload_logger_
                    ->critical("collective callback for unknown online node "
                               "id={}", node_id);
                exit(EXIT_FAILURE);
            }
            if (sys->trace_enabled) {
                workload_logger_
                    ->debug("callback,sys->id={}, tick={}, node->id={}, "
                            "node->name={}, node->type={}",
                            sys->id, Sys::boostedTick(), node_id,
                            nv->name.c_str(), nv->node_type);
            }
            hw_resource->release(*nv);
            stats->record_end(*nv, Sys::boostedTick());
            if (MetricCollector::instance().enabled()) {
                MetricCollector::instance().on_node_complete(
                    sys->id, node_id, Sys::boostedTick());
            }
        } else {
            // Step 1-4: the ETFeederNode handle (static mode) is fetched
            // through the GraphSource.
            node = graph_source_->et_node(node_id);

            if (sys->trace_enabled) {
                workload_logger_
                    ->debug("callback,sys->id={}, tick={}, node->id={}, "
                            "node->name={}, node->type={}",
                            sys->id, Sys::boostedTick(), node->id(),
                            node->name(),
                            static_cast<uint64_t>(node->type()));
            }

            hw_resource->release(node);
            stats->record_end(node, Sys::boostedTick());
            if (MetricCollector::instance().enabled()) {
                MetricCollector::instance().on_node_complete(
                    sys->id, node->id(), Sys::boostedTick());
            }
        }

        // Calculate network bandwidth
        auto& op_stat = stats->get_operator_statistics(node_id);
        Tick execution_time = int_data->execution_time;
        if (execution_time > 0 && op_stat.comm_size.has_value()) {
            double bandwidth =
                static_cast<double>(op_stat.comm_size.value()) / execution_time;
            op_stat.network_bandwidth = bandwidth;
        }

        if (this->sys->track_local_mem) {
            this->local_mem_usage_tracker->recordEnd(node, Sys::boostedTick());
        }

        // Step 1-3: unconditional node-terminal record (collective branch);
        // step 1-5 reverse-index fill in online mode.
        record_node_terminal(graph_source_, execution_mode_, sys->id,
                             node_id,
                             ExecutionDriven::NodeTerminalStatus::Success);

        graph_source_->finish_node(node_id);
        // Static auto-advance only: in online mode the post-commit deferred
        // path drains the store (steps 1-6/1-11); call() must never re-issue.
        if (execution_mode_ == ExecutionDriven::ExecutionMode::Static) {
            issue_dep_free_nodes();
        }

        // The Dataset class provides statistics that should be used later to
        // dump more statistics in the workload layer
        delete collective_comm_wrapper_map[coll_comm_id];
        collective_comm_wrapper_map.erase(coll_comm_id);

    } else {
        if (data == nullptr) {
            // Step 1-8: a bare General event (fire()) is the static
            // auto-advance trigger; online mode never fires it -- the
            // post-commit deferred path calls issue_dep_free_nodes()
            // directly (steps 1-6/1-11). Reaching this in online mode is a
            // mechanism violation: fail closed.
            if (execution_mode_ == ExecutionDriven::ExecutionMode::Online) {
                workload_logger_
                    ->critical("bare General event (fire) reached "
                               "Workload::call in online mode; the "
                               "post-commit path issues dep-free nodes "
                               "directly (step 1-8)");
                exit(EXIT_FAILURE);
            }
            issue_dep_free_nodes();
        } else {
            WorkloadLayerHandlerData* wlhd = (WorkloadLayerHandlerData*)data;
            // Local-HBM contention join: COMM endpoints and charged MEM
            // nodes complete only when both sides arrived. The arrival is
            // consumed (held) when the counterpart is still pending.
            if (try_join_hbm_node(wlhd, event)) {
                // Fall through to the static end-check below (a held
                // arrival never drains the graph: the node stays ongoing).
            } else {
                run_wlhd_terminal(wlhd, event);
                delete wlhd;
            }
        }
    }

    // Final end authority (方案 §4 步骤 1-2 操作 3): in online mode this
    // side effect is disabled -- a temporarily empty graph must never end the
    // online simulation and drop later injections. The end authority belongs
    // exclusively to the ServiceCoordinator (总体方案 §5.5). The static path
    // keeps the pre-phase-1 behavior byte-for-byte.
    if (execution_mode_ != ExecutionDriven::ExecutionMode::Online) {
        // Step 1-4: static_all_done (free empty && ongoing empty) comes from
        // the GraphSource -- the sole dependency-state owner.
        if ((graph_source_->static_all_done()) &&
            (hw_resource->num_in_flight_cpu_ops == 0) &&
            (hw_resource->num_in_flight_gpu_comp_ops == 0) &&
            (hw_resource->num_in_flight_gpu_comm_ops == 0) &&
            // Local-HBM contention: a still-active local-HBM job always
            // belongs to a node that has not completed, so the slot counters
            // above already cover it; the explicit has_active_jobs() check
            // is a belt-and-braces guard -- a drained resolver with pending
            // endpoint jobs must keep waiting for the model's completion
            // callbacks (they re-enter Workload::call and finish the nodes).
            (local_hbm_bandwidth_model == nullptr ||
             !local_hbm_bandwidth_model->has_active_jobs())) {
            report();
            sys->comm_NI->sim_notify_finished();
            is_finished = true;
        }
    }
}

void Workload::arm_hbm_join(uint64_t node_id) {
    // Idempotent by construction: each node issues exactly once and arms at
    // most one join (issue-time call).
    pending_hbm_joins_[node_id] = HbmNodeJoin{};
}

WorkloadLayerHandlerData* Workload::make_hbm_job_wlhd(uint64_t node_id) {
    WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
    wlhd->sys_id = sys->id;
    wlhd->workload = this;
    wlhd->node_id = node_id;
    wlhd->is_local_hbm_job = true;
    return wlhd;
}

bool Workload::try_join_hbm_node(WorkloadLayerHandlerData* wlhd,
                                 EventType event) {
    auto join_it = pending_hbm_joins_.find(wlhd->node_id);
    if (join_it == pending_hbm_joins_.end()) {
        // Fail-closed fire-after guard: every local-HBM endpoint job is
        // issued together with an armed join, so an HBM-side arrival with
        // no live entry means the join already fired (double fire) or the
        // arm was skipped (missing issue) -- falling through to the
        // ordinary terminal would silently finish the node a second time.
        // Network/port-side arrivals legitimately reach this point (nodes
        // without a join complete single-sided).
        if (wlhd->is_local_hbm_job) {
            workload_logger_
                ->critical("HBM job completion for node {} has no pending "
                           "join entry (double fire or missing issue)",
                           wlhd->node_id);
            exit(EXIT_FAILURE);
        }
        return false;
    }
    HbmNodeJoin& join = join_it->second;
    if (wlhd->is_local_hbm_job) {
        // Idempotence guard: the model fires each job exactly once, so a
        // second HBM-side arrival for a live join is a mechanism violation.
        if (join.hbm_done) {
            workload_logger_
                ->critical("duplicate local-HBM job completion for node "
                           "id={}",
                           wlhd->node_id);
            exit(EXIT_FAILURE);
        }
        join.hbm_done = true;
        join.hbm_wlhd = wlhd;
    } else {
        if (join.network_done) {
            workload_logger_
                ->critical("duplicate network-side completion for node "
                           "id={}",
                           wlhd->node_id);
            exit(EXIT_FAILURE);
        }
        join.network_done = true;
        join.held_wlhd = wlhd;
        join.held_event = event;
    }

    if (!(join.network_done && join.hbm_done)) {
        return true;  // hold this side; the counterpart is still pending
    }

    // Both sides arrived: fire the node terminal exactly once. The join
    // record is removed BEFORE the terminal runs so a synchronous
    // re-entry (issue_dep_free_nodes -> issue -> immediate completion)
    // cannot see the stale record.
    WorkloadLayerHandlerData* terminal_wlhd = join.held_wlhd;
    const EventType terminal_event = join.held_event;
    WorkloadLayerHandlerData* hbm_wlhd = join.hbm_wlhd;
    pending_hbm_joins_.erase(join_it);
    run_wlhd_terminal(terminal_wlhd, terminal_event);
    delete terminal_wlhd;
    delete hbm_wlhd;
    return true;
}

void Workload::run_wlhd_terminal(WorkloadLayerHandlerData* wlhd,
                                 EventType event) {
    // Step 1-8: online mode has no ETFeederNode handle (et_node ==
    // nullptr); the online branch releases / records through the
    // NodeView. The static branch below stays byte-identical.
    const ExecutionDriven::NodeView* nv = nullptr;
    shared_ptr<Chakra::FeederV3::ETFeederNode> node = nullptr;
    if (execution_mode_ == ExecutionDriven::ExecutionMode::Online) {
        nv = graph_source_->lookup_ptr(wlhd->node_id);
        if (nv == nullptr) {
            workload_logger_
                ->critical("callback for unknown online node id={}",
                           wlhd->node_id);
            exit(EXIT_FAILURE);
        }
        if (sys->trace_enabled) {
            workload_logger_
                ->debug("callback,sys->id={}, tick={}, node->id={}, "
                        "node->name={}, node->type={}",
                        sys->id, Sys::boostedTick(), wlhd->node_id,
                        nv->name.c_str(), nv->node_type);
        }
        hw_resource->release(*nv);
        stats->record_end(*nv, Sys::boostedTick());
        if (MetricCollector::instance().enabled()) {
            MetricCollector::instance().on_node_complete(
                sys->id, wlhd->node_id, Sys::boostedTick());
        }
        // Phase-4 sensing (方案 §6.2 操作 2): remote-FIFO 实账本层
        // completion record -- the port-FIFO completion callback
        // registers the workload's own event back, so MEM_LOAD/
        // MEM_STORE completions arrive through this generic terminal
        // branch (real-FIFO online path; the replay bypass was
        // deleted with the replay route, 2026-08-18); pure counter
        // observation, no semantics.
        if (nv->kind == ExecutionDriven::NodeKind::MemLoad ||
            nv->kind == ExecutionDriven::NodeKind::MemStore) {
            ExecutionDriven::RemoteFifoLedger::instance()
                .record_completion(sys->id, nv->compute.tensor_size);
        }
    } else {
        node = graph_source_->et_node(wlhd->node_id);

        if (sys->trace_enabled) {
            workload_logger_
                ->debug("callback,sys->id={}, tick={}, node->id={}, "
                        "node->name={}, node->type={}",
                        sys->id, Sys::boostedTick(), node->id(),
                        node->name(),
                        static_cast<uint64_t>(node->type()));
        }

        hw_resource->release(node);
        stats->record_end(node, Sys::boostedTick());
        if (MetricCollector::instance().enabled()) {
            MetricCollector::instance().on_node_complete(
                sys->id, node->id(), Sys::boostedTick());
        }
    }

    // Calculate network bandwidth for point-to-point communications
    if (event == EventType::PacketSent ||
        event == EventType::PacketReceived) {
        auto& op_stat = stats->get_operator_statistics(wlhd->node_id);
        Tick execution_time =
            stats->get_operator_statistics(wlhd->node_id).end_time -
            stats->get_operator_statistics(wlhd->node_id).start_time;
        if (execution_time > 0 && op_stat.comm_size.has_value()) {
            double bandwidth =
                static_cast<double>(op_stat.comm_size.value()) /
                execution_time;
            op_stat.network_bandwidth = bandwidth;
        }
    }

    if (this->sys->track_local_mem) {
        this->local_mem_usage_tracker->recordEnd(node,
                                                 Sys::boostedTick());
    }

    // Step 1-3: unconditional node-terminal record (generic wlhd
    // branch; also reached by AnalyticalRemoteMemory completions,
    // which register_event back into Workload::call, and by the
    // LocalHbmBandwidthModel completion callbacks); step 1-5
    // reverse-index fill in online mode.
    record_node_terminal(graph_source_, execution_mode_, sys->id,
                         wlhd->node_id,
                         ExecutionDriven::NodeTerminalStatus::
                             Success);

    graph_source_->finish_node(wlhd->node_id);
    // Static auto-advance only (see collective branch above).
    if (execution_mode_ == ExecutionDriven::ExecutionMode::Static) {
        issue_dep_free_nodes();
    }
}

void Workload::fire() {
    call(EventType::General, NULL);
}

void Workload::report() {
    Tick curr_tick = Sys::boostedTick();
    workload_logger_
        ->info("sys[{}] finished, {} cycles, exposed communication {} cycles.",
               sys->id, curr_tick, curr_tick - hw_resource->tics_gpu_ops);
    stats->post_processing();
    stats->report();
    if (this->sys->track_local_mem) {
        this->local_mem_usage_tracker->buildMemoryTrace();
        this->local_mem_usage_tracker->buildMemoryTimeline();
        this->local_mem_usage_tracker->dumpMemoryTrace(
            this->sys->local_mem_trace_filename);
        auto [peak_mem_usage, unit] =
            this->local_mem_usage_tracker->getPeakMemUsageFormatted();
        auto logger = workload_logger_;
        logger->info("sys[{}] peak memory usage: {:.2f} {}", sys->id,
                     peak_mem_usage, unit);
        this->local_mem_usage_tracker.reset();
    }
}

CommunicatorGroup* Workload::extract_comm_group(
    const ExecutionDriven::NodeView& node) {
    // pg_name parsed by the GraphSource adapter (legacy "" default).
    std::string comm_group_name = node.coll.pg_name;
    // [default communication group]
    // We assume that an empty comm group, or comm group '0' both correspond
    // to the default communicator group that includes all ranks.
    // If, in the future, we want to support a user-defined comm group '0',
    // revisit this logic.
    if (comm_group_name == "" || comm_group_name == "0") {
        // No communicator group is specified for this communication ET node.
        return nullptr;
    }

    int comm_group_id = std::stoi(comm_group_name);
    if (comm_groups.find(comm_group_id) == comm_groups.end()) {
        workload_logger_
            ->critical(
                "For rank {} ET node {}, communicator group {} not found",
                sys->id, node.global_id, comm_group_id);
        exit(EXIT_FAILURE);
    }
    return comm_groups[comm_group_id];
}
