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

// Step 1-5: node-terminal record with the already-resolved online NodeView.
// Every terminal path that can run online has the view in hand for release /
// stats, so passing it through avoids a second NodeStore hash lookup for every
// terminal. Static mode stays nullptr/0 -- byte-for-byte the pre-phase-1
// shape. The view is alive for the whole call, so its c_str() pointers remain
// valid for the duration of the observer hook.
void record_node_terminal(
    ExecutionDriven::ExecutionMode mode, int rank, uint64_t node_id,
    ExecutionDriven::NodeTerminalStatus status,
    const ExecutionDriven::NodeView* online_node) {
    const ExecutionDriven::NodeView* const nv =
        mode == ExecutionDriven::ExecutionMode::Online ? online_node : nullptr;
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
    this->hw_resource = new HardwareResource(1, sys->id, execution_mode);
    this->local_mem_usage_tracker =
        std::make_unique<LocalMemUsageTracker>(sys->id);
    // Multi-user local-HBM bandwidth contention: one fluid model per rank
    // when the flag is on (Sys force-disables it for local_mem_bw <= 0).
    // The model only interacts with this rank's jobs; NoC multi-hop routes
    // never create jobs on intermediate ranks (endpoint-only charging).
    if (sys->hbm_bandwidth_contention) {
        this->local_hbm_bandwidth_model =
            std::make_unique<LocalHbmBandwidthModel>(sys, this);
    }
    this->sys = sys;
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
    if (this->execution_mode_ == ExecutionDriven::ExecutionMode::Online) {
        this->stats->configure_online_history_preservation(
            MetricCollector::instance().preserve_online_operator_history());
    }
    this->is_finished = false;
}

Workload::~Workload() {
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

ExecutionDriven::OnlineStatisticsState&
Workload::online_statistics_state_or_fail(uint64_t node_id) {
    auto* state = graph_source_->mutable_online_statistics(node_id);
    if (state == nullptr) {
        workload_logger_->critical(
            "compact online statistics state missing for node {}", node_id);
        std::exit(EXIT_FAILURE);
    }
    return *state;
}

void Workload::start_online_statistics(
    const ExecutionDriven::NodeView& node, Tick start_time) {
    if (stats->online_history_preserved()) {
        stats->record_start(node, start_time);
        return;
    }
    stats->record_online_service_start(
        node, online_statistics_state_or_fail(node.global_id), start_time);
}

void Workload::complete_online_statistics(
    const ExecutionDriven::NodeView& node, Tick end_time) {
    if (stats->online_history_preserved()) {
        stats->record_end(node, end_time);
        return;
    }
    stats->complete_online_service_operator(
        node, online_statistics_state_or_fail(node.global_id), end_time);
}

void Workload::mark_online_terminal_or_fail(uint64_t node_id) {
    if (execution_mode_ != ExecutionDriven::ExecutionMode::Online) {
        return;
    }
    if (!graph_source_->mark_terminal_observed(node_id)) {
        workload_logger_->critical(
            "duplicate, unknown, or unissued online terminal callback for "
            "node {}",
            node_id);
        std::exit(EXIT_FAILURE);
    }
}

void Workload::record_network_bandwidth(uint64_t node_id,
                                        Tick execution_time) {
    if (execution_mode_ == ExecutionDriven::ExecutionMode::Online &&
        !stats->online_history_preserved()) {
        auto& online_stat = online_statistics_state_or_fail(node_id);
        if (execution_time > 0 && online_stat.comm_size.has_value()) {
            online_stat.network_bandwidth =
                static_cast<double>(online_stat.comm_size.value()) /
                execution_time;
        }
        return;
    }
    auto& op_stat = stats->get_operator_statistics(node_id);
    if (execution_time > 0 && op_stat.comm_size.has_value()) {
        op_stat.network_bandwidth =
            static_cast<double>(op_stat.comm_size.value()) / execution_time;
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

        comm_groups[comm_group_id] = std::make_shared<CommunicatorGroup>(
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
            auto existing = this->comm_groups.find(pgNameInt);
            if (existing != this->comm_groups.end() && existing->second &&
                existing->second->get_id() == pgNameInt + 1 &&
                existing->second->matches_definition(involved_NPUs)) {
                continue;
            }
            // To ensure pgName > 0
            auto cg = std::make_shared<CommunicatorGroup>(
                pgNameInt + 1, involved_NPUs, sys);
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
            // real serialized physics (path-2 removal 2026-08-18: the
            // replay-only concurrent calibrated-COMP bypass was deleted with
            // the replay route).
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
    std::shared_ptr<Chakra::FeederV3::ETFeederNode> et_node = nullptr;
    if (execution_mode_ == ExecutionDriven::ExecutionMode::Online) {
        this->hw_resource->occupy(node);
        // stats->record_end will be called in Workload::call
        start_online_statistics(node, Sys::boostedTick());
    } else {
        et_node = graph_source_->et_node(node.global_id);
        this->hw_resource->occupy(et_node);
        // stats->record_end will be called in Workload::call
        stats->record_start(et_node, Sys::boostedTick());
    }
    // Side-band metrics observation only; does not touch the node, the
    // dependency resolver, or the event queue (doc sec.5.5).
    // R2 (2026-08-29) anchor fast path: online mode consults the sparse
    // NodeView flag set by the B-1.5 registration hook; unanchored nodes
    // (the overwhelming majority) skip the two-level hash lookup entirely.
    // Static/ET mode keeps the unconditional call -- ETFeederGraphSource
    // views leave the flags false, and the static manifest anchor
    // semantics are a red line.
    if (MetricCollector::instance().enabled() &&
        (execution_mode_ != ExecutionDriven::ExecutionMode::Online ||
         node.metric_issue_anchor)) {
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
    // R3 (方案 §3.6 / 阶段 E, 2026-08-29): the remote-FIFO ledger issue
    // record moved INTO the backend -- AnalyticalRemoteMemory::issue counts
    // on the real resolved port_index (Workload must not infer a port from
    // sys_id; see RemoteFifoLedger.hh). The call below is the only issue
    // call site in the repo, so per-request accounting coverage is
    // unchanged, now with the correct key under every memory architecture.
    sys->remote_mem->issue(node.compute.tensor_size, wlhd);
}

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
    wlhd->node_id = node.global_id;

    const uint64_t node_num_ops = node.compute.num_ops;
    const uint64_t node_tensor_size = node.compute.tensor_size;
    double num_ops = static_cast<double>(node_num_ops);
    double tensor_size = static_cast<double>(node_tensor_size);

    // if tensor_size is 0 during roofline mode, this is an invalid node
    if (tensor_size == 0) {
        delete wlhd;
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
    if (local_hbm_bandwidth_model != nullptr &&
        node.compute.runtime_ns == 0ul) {
        // HBM bandwidth contention: COMP executes through the per-rank fluid
        // model (bytes = tensor_size, read+write merged; FLOPs drain at
        // peak_perf in parallel; both drains must finish, preserving
        // Roofline's max() semantics for a single user).  A non-zero
        // calibrated runtime_ns never enters the model and keeps its
        // calibrated duration (this repository's traces always take the
        // fluid path; the calibrated branch is the online-LUT honor path).
        // tics_gpu_ops is accumulated by the model at completion.  Falls
        // through to the shared operator-statistics tail below.
        wlhd->sys_id = sys->id;
        wlhd->workload = this;
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

    if (execution_mode_ == ExecutionDriven::ExecutionMode::Online &&
        !stats->online_history_preserved()) {
        auto& online_stat = online_statistics_state_or_fail(node.global_id);
        online_stat.operation_intensity = operational_intensity;
        online_stat.compute_utilization = perf / sys->peak_perf;
        online_stat.memory_utilization =
            (perf / operational_intensity) / sys->local_mem_bw;
        online_stat.is_memory_bound = perf < sys->peak_perf;
        if (sys->trace_enabled) {
            workload_logger_
                ->debug("operation_intensity={}, perf={}, elapsed_time={} "
                        "local_mem_latency_ns={} "
                        "compute_utilization={} memory_utilization={} tensor_size={} "
                        "num_ops={}",
                        operational_intensity, perf, elapsed_time,
                        sys->local_mem_latency,
                        online_stat.compute_utilization.value(),
                        online_stat.memory_utilization.value(), tensor_size,
                        num_ops);
        }
    } else {
        // Static ET and history-preserving online microbenchmarks retain the
        // legacy complete per-node Statistics record.
        auto& op_stat = this->stats->get_operator_statistics(node.global_id);
        op_stat.operation_intensity = operational_intensity;
        op_stat.compute_utilization = perf / sys->peak_perf;
        op_stat.memory_utilization =
            (perf / operational_intensity) / sys->local_mem_bw;
        op_stat.is_memory_bound = perf < sys->peak_perf;
        if (sys->trace_enabled) {
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
    }
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

    auto comm_group_owner = extract_comm_group(node);
    CommunicatorGroup* comm_group = comm_group_owner.get();
    const auto comm_type =
        static_cast<ChakraCollectiveCommType>(node.coll.comm_type);
    const auto comm_size = node.coll.bytes;
    // Keep comm_size on the live NodeStore record in compact service mode:
    // terminal bandwidth accounting still consumes it, but no global
    // Statistics per-node hash-table entry is needed.
    if (execution_mode_ == ExecutionDriven::ExecutionMode::Online &&
        !stats->online_history_preserved()) {
        online_statistics_state_or_fail(node.global_id).comm_size = comm_size;
    } else {
        stats->get_operator_statistics(node.global_id).comm_size = comm_size;
    }
    // TODO: comm_tag? which is used to distinguish two different collective in
    // same pg
    const auto comm_priority = node.coll.priority;  // default 0u

    if (comm_type == ChakraCollectiveCommType::ALL_REDUCE) {
        DataSet* fp = sys->generate_all_reduce(comm_size, involved_dims,
                                               comm_group, comm_priority, node.global_id);
        fp->retain_communicator_group(comm_group_owner);
        collective_comm_node_id_map[fp->my_id] = node.global_id;
        collective_comm_wrapper_map[fp->my_id] = fp;
        fp->set_notifier(this, EventType::CollectiveCommunicationFinished);
    } else if (comm_type == ChakraCollectiveCommType::ALL_TO_ALL) {
        DataSet* fp = sys->generate_all_to_all(comm_size, involved_dims,
                                               comm_group, comm_priority, node.global_id);
        fp->retain_communicator_group(comm_group_owner);
        collective_comm_node_id_map[fp->my_id] = node.global_id;
        collective_comm_wrapper_map[fp->my_id] = fp;
        fp->set_notifier(this, EventType::CollectiveCommunicationFinished);
    } else if (comm_type == ChakraCollectiveCommType::ALL_GATHER) {
        DataSet* fp = sys->generate_all_gather(comm_size, involved_dims,
                                               comm_group, comm_priority, node.global_id);
        fp->retain_communicator_group(comm_group_owner);
        collective_comm_node_id_map[fp->my_id] = node.global_id;
        collective_comm_wrapper_map[fp->my_id] = fp;
        fp->set_notifier(this, EventType::CollectiveCommunicationFinished);
    } else if (comm_type == ChakraCollectiveCommType::REDUCE_SCATTER) {
        DataSet* fp = sys->generate_reduce_scatter(comm_size, involved_dims,
                                                   comm_group, comm_priority, node.global_id);
        fp->retain_communicator_group(comm_group_owner);
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
        fp->retain_communicator_group(comm_group_owner);
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
    // Record communication size for bandwidth calculation.
    if (execution_mode_ == ExecutionDriven::ExecutionMode::Online &&
        !stats->online_history_preserved()) {
        online_statistics_state_or_fail(node.global_id).comm_size = size;
    } else {
        stats->get_operator_statistics(node.global_id).comm_size = size;
    }
    const auto tag = node.comm.tag;

    if (local_hbm_bandwidth_model != nullptr && size > 0 &&
        node.comm.hbm_charge) {
        // HBM bandwidth contention, sending endpoint: this rank READS the
        // payload out of its local HBM while the fluid NoC transfer runs.
        // The node completes at the join of the network-side callback and
        // the endpoint HBM job (see HbmCommJoin); the job starts now, at
        // the node issue tick.  Intermediate ranks on a multi-hop route
        // never create jobs (router pass-through: only endpoints charge
        // HBM).  bytes == 0 and hbm-charge == false stay on the plain
        // network-only completion path.
        WorkloadLayerHandlerData* hbm_wlhd = new WorkloadLayerHandlerData;
        hbm_wlhd->sys_id = sys->id;
        hbm_wlhd->workload = this;
        hbm_wlhd->node_id = node.global_id;
        hbm_comm_join_[node.global_id] =
            HbmCommJoin{false, false, EventType::PacketSent};
        local_hbm_bandwidth_model->issue_comm_read(size, hbm_wlhd);
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
    // Record communication size for bandwidth calculation.
    if (execution_mode_ == ExecutionDriven::ExecutionMode::Online &&
        !stats->online_history_preserved()) {
        online_statistics_state_or_fail(node.global_id).comm_size = size;
    } else {
        stats->get_operator_statistics(node.global_id).comm_size = size;
    }
    const auto tag = node.comm.tag;

    if (local_hbm_bandwidth_model != nullptr && size > 0 &&
        node.comm.hbm_charge) {
        // HBM bandwidth contention, receiving endpoint: this rank WRITES the
        // incoming payload into its local HBM.  Same join semantics as the
        // send side (issue_send_comm); the job is created before the recv is
        // posted so that an already-finished transmission (immediate network
        // callback inside front_end_sim_recv) still finds a pending join
        // entry instead of completing the node outright.
        WorkloadLayerHandlerData* hbm_wlhd = new WorkloadLayerHandlerData;
        hbm_wlhd->sys_id = sys->id;
        hbm_wlhd->workload = this;
        hbm_wlhd->node_id = node.global_id;
        hbm_comm_join_[node.global_id] =
            HbmCommJoin{false, false, EventType::PacketReceived};
        local_hbm_bandwidth_model->issue_comm_write(size, hbm_wlhd);
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
    // Claim terminal delivery before any observer/statistics side effect.
    // Static mode is intentionally a no-op; online duplicate callbacks are a
    // mechanism error rather than an idempotent second aggregate.
    mark_online_terminal_or_fail(node_id);
    // Step 1-3: unconditional node-terminal record (独立于 metrics 开关).
    // Step 1-5: the online path fills the reverse index from the GraphSource
    // view; static stays nullptr/0. Skipped is explicit: the node never
    // executed (INVALID_NODE / metadata / zero tensor_size); whether Skipped
    // satisfies a watch is the watch's own policy, never a default.
    record_node_terminal(execution_mode_, sys->id, node_id,
                         ExecutionDriven::NodeTerminalStatus::Skipped, &node);
    // Step 1-4: the GraphSource is the sole dependency-state owner.
    graph_source_->finish_node(node_id);
    auto logger = workload_logger_;
    // C4 (2026-08-28): gate the per-node debug format behind trace_enabled
    // (same pattern as issue()) -- with trace off the format string and
    // arguments are not even evaluated, and skip_invalid runs once per
    // invalid node in online mode.
    if (sys->trace_enabled) {
        logger->debug("callback,sys->id={}, tick={}, node->id={}, "
                      "node->name={}, node->type={}",
                      sys->id, Sys::boostedTick(), node.global_id, node.name,
                      static_cast<uint64_t>(node.node_type));
    }
    // Step 1-8: online path (et_node == nullptr) releases through the
    // NodeView overloads. The track_local_mem block below keeps the static
    // et_node handle (unreachable in online mode: the ctor fails closed on
    // track_local_mem + Online).
    std::shared_ptr<Chakra::FeederV3::ETFeederNode> et_node = nullptr;
    if (execution_mode_ == ExecutionDriven::ExecutionMode::Online) {
        // `node` is the GraphSource view just taken by issue(); do not look
        // it up again on this all-node terminal path.
        hw_resource->release(node);
        complete_online_statistics(node, Sys::boostedTick());
    } else {
        et_node = graph_source_->et_node(node_id);
        hw_resource->release(et_node);
        stats->record_end(et_node, Sys::boostedTick());
    }
    // R2 (2026-08-29) anchor fast path: online mode consults the sparse
    // NodeView flag (set at B-1.5 registration); static/ET mode keeps the
    // unconditional call (skip_invalid is shared by both modes, and the
    // ETFeeder views' flags are always false).
    if (MetricCollector::instance().enabled() &&
        (execution_mode_ != ExecutionDriven::ExecutionMode::Online ||
         node.metric_complete_anchor)) {
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

        auto node_id_it = collective_comm_node_id_map.find(coll_comm_id);
        auto wrapper_it = collective_comm_wrapper_map.find(coll_comm_id);
        if (node_id_it == collective_comm_node_id_map.end() ||
            wrapper_it == collective_comm_wrapper_map.end() ||
            wrapper_it->second == nullptr) {
            workload_logger_
                ->critical("collective callback for missing or already "
                           "retired dataset id={}", coll_comm_id);
            exit(EXIT_FAILURE);
        }
        const uint64_t node_id = node_id_it->second;
        DataSet* collective_wrapper = wrapper_it->second;
        collective_comm_node_id_map.erase(node_id_it);
        collective_comm_wrapper_map.erase(wrapper_it);

        hw_resource->tics_gpu_comms += int_data->execution_time;
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
            mark_online_terminal_or_fail(node_id);
            hw_resource->release(*nv);
        complete_online_statistics(*nv, Sys::boostedTick());
            // R2 anchor fast path: online-only branch (the static else below
            // is untouched and stays unconditional); nv points into the
            // NodeStore record the registration hook set the flags on.
            if (MetricCollector::instance().enabled() &&
                nv->metric_complete_anchor) {
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
        record_network_bandwidth(node_id, int_data->execution_time);

        if (this->sys->track_local_mem) {
            this->local_mem_usage_tracker->recordEnd(node, Sys::boostedTick());
        }

        // Step 1-3: unconditional node-terminal record (collective branch);
        // step 1-5 reverse-index fill in online mode.
        record_node_terminal(execution_mode_, sys->id, node_id,
                             ExecutionDriven::NodeTerminalStatus::Success, nv);

        graph_source_->finish_node(node_id);
        // Static auto-advance only: in online mode the post-commit deferred
        // path drains the store (steps 1-6/1-11); call() must never re-issue.
        if (execution_mode_ == ExecutionDriven::ExecutionMode::Static) {
            issue_dep_free_nodes();
        }

        delete collective_wrapper;

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
            if ((event == EventType::PacketSent ||
                 event == EventType::PacketReceived) &&
                hbm_comm_join_.find(wlhd->node_id) != hbm_comm_join_.end()) {
                // HBM-joined p2p comm node, network side: only record the
                // arrival; the node completes when the endpoint HBM job has
                // also finished (maybe_complete_hbm_joined_comm below).  The
                // network payload (wlhd) is not needed for the final
                // completion, which is driven by the join entry.
                const uint64_t joined_node_id = wlhd->node_id;
                hbm_comm_join_[joined_node_id].network_done = true;
                delete wlhd;
                maybe_complete_hbm_joined_comm(joined_node_id);
            } else {
                finish_generic_node(wlhd->node_id, event);
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

void Workload::fire() {
    call(EventType::General, NULL);
}

void Workload::finish_generic_node(uint64_t node_id, EventType event) {
    // Extracted verbatim from the former generic wlhd branch of
    // Workload::call: node release / stats / metrics / terminal record /
    // dependency release / static-mode auto-advance.  Shared by the plain
    // register_event completions, the LocalHbmBandwidthModel COMP
    // completions, and the joined p2p comm completions.
    const ExecutionDriven::NodeView* nv = nullptr;
    shared_ptr<Chakra::FeederV3::ETFeederNode> node = nullptr;
    if (execution_mode_ == ExecutionDriven::ExecutionMode::Online) {
        nv = graph_source_->lookup_ptr(node_id);
        if (nv == nullptr) {
            workload_logger_
                ->critical("callback for unknown online node id={}",
                           node_id);
            exit(EXIT_FAILURE);
        }
        if (sys->trace_enabled) {
            workload_logger_
                ->debug("callback,sys->id={}, tick={}, node->id={}, "
                        "node->name={}, node->type={}",
                        sys->id, Sys::boostedTick(), node_id,
                        nv->name.c_str(), nv->node_type);
        }
        mark_online_terminal_or_fail(node_id);
        hw_resource->release(*nv);
        complete_online_statistics(*nv, Sys::boostedTick());
        // R2 anchor fast path: online-only branch (the static else below is
        // untouched and stays unconditional); nv points into the NodeStore
        // record the registration hook set the flags on.
        if (MetricCollector::instance().enabled() &&
            nv->metric_complete_anchor) {
            MetricCollector::instance().on_node_complete(
                sys->id, node_id, Sys::boostedTick());
        }
        // R3 (方案 §3.6 / 阶段 E, 2026-08-29): the remote-FIFO ledger
        // completion record moved INTO the backend --
        // AnalyticalRemoteMemory::call counts from its own completion
        // payload (this transaction's port AND bytes). Accounting here
        // had two defects: the key was sys_id (a shared port's queue was
        // split into per-rank virtual ledgers) and the record moment was
        // the node terminal, which an HBM join can hold past the real
        // port-transaction completion.
    } else {
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

    // Calculate network bandwidth for point-to-point communications.  For a
    // joined comm node this runs at the join completion (max of the network
    // and HBM endpoint times), so the reported bandwidth already reflects
    // the HBM contention delay -- the endpoint is part of the transfer.
    if (event == EventType::PacketSent || event == EventType::PacketReceived) {
        if (execution_mode_ == ExecutionDriven::ExecutionMode::Online &&
            !stats->online_history_preserved()) {
            const auto& online_stat = online_statistics_state_or_fail(node_id);
            if (!online_stat.completed ||
                online_stat.end_time ==
                    ExecutionDriven::OnlineStatisticsState::kInvalidTick) {
                workload_logger_->critical(
                    "p2p bandwidth requested before compact online completion "
                    "for node {}",
                    node_id);
                std::exit(EXIT_FAILURE);
            }
            record_network_bandwidth(
                node_id, online_stat.end_time - online_stat.start_time);
        } else {
            const auto& op_stat = stats->get_operator_statistics(node_id);
            record_network_bandwidth(
                node_id, op_stat.end_time - op_stat.start_time);
        }
    }

    if (this->sys->track_local_mem) {
        this->local_mem_usage_tracker->recordEnd(node, Sys::boostedTick());
    }

    // Step 1-3: unconditional node-terminal record (generic wlhd branch;
    // also reached by AnalyticalRemoteMemory completions, which
    // register_event back into Workload::call); step 1-5 reverse-index fill
    // in online mode.
    record_node_terminal(execution_mode_, sys->id, node_id,
                         ExecutionDriven::NodeTerminalStatus::Success, nv);

    graph_source_->finish_node(node_id);
    // Static auto-advance only (see the collective branch in Workload::call).
    if (execution_mode_ == ExecutionDriven::ExecutionMode::Static) {
        issue_dep_free_nodes();
    }
}

void Workload::maybe_complete_hbm_joined_comm(uint64_t node_id) {
    auto it = hbm_comm_join_.find(node_id);
    if (it == hbm_comm_join_.end()) {
        return;  // already fired (idempotence guard) or not a joined node
    }
    if (!(it->second.network_done && it->second.hbm_done)) {
        return;  // still waiting for the other side
    }
    const EventType completion_event = it->second.completion_event;
    // Erase BEFORE completing: the completion re-enters the issue path
    // (static auto-advance) and any re-entrant lookup for this node must
    // observe an already-fired join.
    hbm_comm_join_.erase(it);
    finish_generic_node(node_id, completion_event);
}

void Workload::on_local_hbm_job_complete(
    WorkloadLayerHandlerData* wlhd,
    LocalHbmBandwidthModel::JobKind kind) {
    const uint64_t node_id = wlhd->node_id;
    if (kind == LocalHbmBandwidthModel::JobKind::COMM_READ ||
        kind == LocalHbmBandwidthModel::JobKind::COMM_WRITE) {
        auto it = hbm_comm_join_.find(node_id);
        if (it == hbm_comm_join_.end()) {
            workload_logger_
                ->critical("HBM job completion for node {} has no pending "
                           "join entry (double fire or missing issue)",
                           node_id);
            exit(EXIT_FAILURE);
        }
        it->second.hbm_done = true;
        delete wlhd;
        maybe_complete_hbm_joined_comm(node_id);
        return;
    }
    // COMP job: single-event completion, same path as the closed-form
    // register_event route.
    finish_generic_node(node_id, EventType::General);
    delete wlhd;
}

void Workload::report() {
    // Compact online service has no legacy per-node Statistics history. Check
    // before emitting any report output so this path cannot report zeros.
    stats->ensure_legacy_post_processing_supported();
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

std::shared_ptr<CommunicatorGroup> Workload::extract_comm_group(
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
    const auto comm_group_it = comm_groups.find(comm_group_id);
    if (comm_group_it == comm_groups.end() || !comm_group_it->second) {
        workload_logger_
            ->critical(
                "For rank {} ET node {}, communicator group {} not found",
                sys->id, node.global_id, comm_group_id);
        exit(EXIT_FAILURE);
    }
    return comm_group_it->second;
}
