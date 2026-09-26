/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/workload/Workload.hh"

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/IntData.hh"
#include "astra-sim/system/RecvPacketEventHandlerData.hh"
#include "astra-sim/system/SendPacketEventHandlerData.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include "astra-sim/workload/LocalHbmBandwidthModel.hh"
#include "astra-sim/workload/MetricCollector.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include <json/json.hpp>

#include <algorithm>
#include <cmath>
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
    this->sys = sys;
    if (sys->hbm_bandwidth_contention || sys->hbm_kv_restore_bandwidth_sharing) {
        // sh_2.0: the LocalHbmBandwidthModel is a preserved execution model
        // (contract B11) -- assembled in BOTH modes, identical timing/params.
        // With hbm-bandwidth-contention (default) it runs the N-way
        // equal-split policy over every HBM user (COMP, KV restore, NoC p2p
        // comm endpoints, pool endpoints); with only the legacy
        // hbm-kv-restore-bandwidth-sharing flag (contention off) no comm/pool
        // jobs are ever issued, so the model degenerates to the historical
        // one-COMP-plus-one-restore 50/50 behavior (A/B baseline).
        //
        // The model constructor requires positive local-mem-bw AND
        // peak-perf. Sys force-disables hbm-bandwidth-contention for
        // local-mem-bw <= 0, but that guard does not cover
        // hbm-kv-restore-bandwidth-sharing, and a missing "peak-perf" key
        // defaults peak_perf to 0. Fail closed with a clear diagnostic
        // instead of letting std::invalid_argument escape this constructor
        // (it runs under "new Workload" inside Sys, with no try/catch up
        // the chain, so an escape is std::terminate at startup).
        try {
            this->local_hbm_bandwidth_model =
                std::make_unique<LocalHbmBandwidthModel>(sys, this);
        } catch (const std::invalid_argument& e) {
            workload_logger_->critical(
                "local HBM bandwidth model requested "
                "(hbm-bandwidth-contention={}, "
                "hbm-kv-restore-bandwidth-sharing={}) but its parameters "
                "are missing or non-positive in the system configuration "
                "(local-mem-bw={}, peak-perf={}): {}",
                sys->hbm_bandwidth_contention,
                sys->hbm_kv_restore_bandwidth_sharing, sys->local_mem_bw,
                sys->peak_perf, e.what());
            exit(EXIT_FAILURE);
        }
    }
    this->comm_groups.clear();
    // TODO: parametrize the number of available hardware resources
    this->hw_resource = new HardwareResource(sys->id, execution_mode);
    this->local_mem_usage_tracker =
        std::make_unique<LocalMemUsageTracker>(sys->id);
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

// record_network_bandwidth was removed: the whole comm_size ->
// achieved-bandwidth chain had no production consumer (the former
// OperatorStatistics::network_bandwidth field was write-only, its only
// reader a long-commented-out report block; the field itself is gone).
// OperatorStatistics::comm_size was write-only for this chain and is
// deleted with it; the compact-mode OnlineStatisticsState::comm_size
// retention stays (live NodeStore record fact, pinned by
// statistics_online_compaction_test).

void Workload::initialize_comm_groups(string comm_group_filename) {
    // communicator group input file is not given
    if (comm_group_filename.find("empty") != std::string::npos) {
        comm_groups.clear();
        return;
    }

    // Fail closed with a diagnostic instead of an escaping exception: this
    // runs in the Workload constructor ("new Workload" inside Sys, no
    // try/catch up the chain), so a missing/unreadable file, a malformed
    // document, or a non-numeric group key would otherwise surface as a
    // bare std::terminate at startup.
    ifstream inFile;
    json j;
    inFile.open(comm_group_filename);
    if (!inFile.is_open()) {
        workload_logger_->critical(
            "communicator group file: {} does not exist or is not readable",
            comm_group_filename);
        exit(EXIT_FAILURE);
    }
    try {
        inFile >> j;
    } catch (const std::exception& e) {
        workload_logger_->critical(
            "failed to parse communicator group file {}: {}",
            comm_group_filename, e.what());
        exit(EXIT_FAILURE);
    }

    for (json::iterator it = j.begin(); it != j.end(); ++it) {
        std::string comm_group_name = it.key();
        // Group keys are decimal ids; reject non-numeric or
        // trailing-garbage keys explicitly (std::stoi would throw, same
        // rule as the ParsedGraphBatch watch-member rank keys).
        int comm_group_id = 0;
        try {
            size_t parsed_chars = 0;
            comm_group_id = std::stoi(comm_group_name, &parsed_chars);
            if (parsed_chars != comm_group_name.size()) {
                workload_logger_->critical(
                    "communicator group file {}: unparsable group id key "
                    "{}, trailing characters",
                    comm_group_filename, comm_group_name);
                exit(EXIT_FAILURE);
            }
        } catch (const std::exception&) {
            workload_logger_->critical(
                "communicator group file {}: unparsable group id key {}",
                comm_group_filename, comm_group_name);
            exit(EXIT_FAILURE);
        }

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
            workload_logger_->critical(
                "Communicator group must be a rank array or an object with "
                "ranks and dimensions (file {} key {})",
                comm_group_filename, comm_group_name);
            exit(EXIT_FAILURE);
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
            if (node.is_local_hbm_kv_restore) {
                issue_local_hbm_kv_restore(node);
            } else {
                issue_remote_mem(node);
            }
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
    issue_pytorch_pg_metadata(node);
    this->skip_invalid(node);  // for proper dependancy resolving
}

void Workload::issue_replay(const ExecutionDriven::NodeView& node) {
    WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
    wlhd->sys_id = sys->id;
    wlhd->workload = this;
    wlhd->node_id = node.global_id;
    uint64_t runtime = 1ul;
    if (node.compute.runtime_ns != 0ul) {
        // chakra runtimes are in microseconds and the GraphSource adapter
        // already converted them into nanoseconds
        runtime = node.compute.runtime_ns;
    }
    if (!node.is_cpu_op) {
        hw_resource->tics_gpu_ops += runtime;
    }
    sys->register_event(this, EventType::General, wlhd, runtime);
}

void Workload::issue_remote_mem(const ExecutionDriven::NodeView& node) {
    WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
    wlhd->sys_id = sys->id;
    wlhd->workload = this;
    wlhd->node_id = node.global_id;
    // sh_2.0 N-way HBM contention: a MEM node marked hbm-access-mode (1 =
    // local HBM read, 2 = local HBM write; bytes = tensor_size) is a pool
    // traffic endpoint -- its local HBM job competes in the N-way model and
    // the node completes as the join of the port FIFO transaction and the
    // HBM job (hbm_endpoint_joins_ latch, see Workload::call).
    if (sys->hbm_bandwidth_contention &&
        local_hbm_bandwidth_model != nullptr &&
        node.hbm_access_mode > 0 && node.compute.tensor_size > 0) {
        hbm_endpoint_joins_[node.global_id] = HbmEndpointJoinState{};
        if (node.hbm_access_mode == 1) {
            local_hbm_bandwidth_model->issue_pool_read(
                node.compute.tensor_size, wlhd);
        } else {
            local_hbm_bandwidth_model->issue_pool_write(
                node.compute.tensor_size, wlhd);
        }
    }
    // Path-2 removal (2026-08-18): the replay-only instant (1ns) remote MEM
    // completion branch was deleted with the replay route; strategy/static
    // keep the real remote FIFO physics.
    sys->remote_mem->issue(node.compute.tensor_size, wlhd);
}

void Workload::issue_local_hbm_kv_restore(
    const ExecutionDriven::NodeView& node) {
    WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
    wlhd->sys_id = sys->id;
    wlhd->workload = this;
    wlhd->node_id = node.global_id;

    const uint64_t tensor_size = node.compute.tensor_size;
    // Side-band restore byte accounting (doc sec.5.2/9.4); the runtime model
    // counters in LocalHbmBandwidthModel track the same bytes as served.
    if (MetricCollector::instance().enabled()) {
        MetricCollector::instance().on_local_hbm_restore_issue(sys->id,
                                                               tensor_size);
    }
    // Zero-byte restore guard: a tensor_size==0 restore never creates an HBM
    // job (the same "bytes == 0 不建作业" tolerance the comm/pool endpoints
    // have, and the same behavior the closed-form fallback below already
    // gives). Both configurations complete it as a pure-latency General
    // event, so the N-way model's fail-closed zero-byte rejection stays a
    // wiring bug, never a data property.
    if (tensor_size == 0) {
        sys->register_event(
            this, EventType::General, wlhd,
            std::max<uint64_t>(
                1, static_cast<uint64_t>(sys->local_mem_latency)));
        return;
    }
    // Path-2 removal (2026-08-18): the replay-only instant (1ns) HBM restore
    // completion branch was deleted with the replay route; strategy/static
    // keep the real LocalHbmBandwidthModel DMA physics.
    if (local_hbm_bandwidth_model != nullptr) {
        local_hbm_bandwidth_model->issue_restore(tensor_size, wlhd);
        return;
    }

    // Legacy closed-form fallback (both HBM flags off). A non-positive
    // local-mem-bw would make the division below Inf, ceil into a huge
    // uint64 runtime, and hang the simulation: fail closed instead.
    if (sys->local_mem_bw <= 0) {
        workload_logger_->critical(
            "local HBM KV-restore fallback requires a positive local-mem-bw "
            "in the system configuration (local-mem-bw={}, "
            "hbm-bandwidth-contention={}, "
            "hbm-kv-restore-bandwidth-sharing={})",
            sys->local_mem_bw, sys->hbm_bandwidth_contention,
            sys->hbm_kv_restore_bandwidth_sharing);
        exit(EXIT_FAILURE);
    }
    const double elapsed_seconds =
        (static_cast<double>(sys->local_mem_latency) / 1e9) +
        static_cast<double>(tensor_size) / sys->local_mem_bw;
    const uint64_t runtime = std::max<uint64_t>(
        1, static_cast<uint64_t>(std::ceil(elapsed_seconds * 1e9)));
    sys->register_event(this, EventType::General, wlhd, runtime);
}

void Workload::issue_comp(const ExecutionDriven::NodeView& node) {
    // issue() dispatch chain: the node was already taken/occupied by the
    // time we run, and static-path callers run under Sys::call_events,
    // which swallows std::exception and leaves the node occupied forever
    // (static_all_done never fires). Any invariant breach here must be
    // fail-closed (critical + exit), not throw.
    if (!this->sys->roofline_enabled) {
        workload_logger_->critical(
            "Roofline model is not enabled for non-replay comp");
        exit(EXIT_FAILURE);
    }

    if (node.is_cpu_op) {
        workload_logger_->critical(
            "Roofline is only available for GPU nodes");
        exit(EXIT_FAILURE);
    }

    // Fail-closed configuration check: roofline is enabled, but its two
    // divisors below were never configured (Sys defaults both to 0 when the
    // "peak-perf"/"local-mem-bw" keys are missing or non-positive).  Using
    // them would produce NaN/Inf perf or utilization values.
    if (sys->peak_perf <= 0.0 || sys->local_mem_bw <= 0.0) {
        workload_logger_->critical(
            "roofline is enabled but peak-perf/local-mem-bw are missing or "
            "non-positive in the system configuration (peak_perf={}, "
            "local_mem_bw={})",
            sys->peak_perf, sys->local_mem_bw);
        exit(EXIT_FAILURE);
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

    // A zero-FLOP node has no roofline value at all; skip the divisions and
    // keep benign zeros instead of 0/0 = NaN (perf is 0 at zero intensity).
    double operational_intensity = 0.0;
    double perf = 0.0;
    double compute_elapsed_time = 0.0;  // sec
    if (node_num_ops != 0ul) {
        operational_intensity = num_ops / tensor_size;
        perf = sys->roofline->get_perf(operational_intensity);
        compute_elapsed_time = num_ops / perf;  // sec
    }
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
            workload_logger_->critical(
                "HBM KV-restore sharing cannot be combined with remote "
                "operand pipeline loads on the same COMP node");
            exit(EXIT_FAILURE);
        }
        if (sys->remote_mem_bw <= 0) {
            workload_logger_->critical(
                "Pipeline roofline requires remote-mem-bw in system config");
            exit(EXIT_FAILURE);
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

    // sec -> ns, rounded up with a 1ns floor -- the same conversion
    // convention as the restore closed-form fallback above and the HBM
    // model's fluid transition scheduler (never a 0ns event).
    uint64_t runtime = std::max<uint64_t>(
        1, static_cast<uint64_t>(std::ceil(elapsed_time * 1e9)));
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
        // sh_2.0: HBM 50/50 fluid sharing executes COMP through the model
        // (compute job), self-timed; the register_event path below is the
        // no-sharing fallback (byte-exact static behavior). The online
        // calibrated runtime_ns branch (GraphBatch planner-LUT-aligned
        // durations) takes the plain register_event path so the engine
        // timeline is order-isomorphic to the replay decision log.
        // Falls through to the shared operator-statistics tail below.
        local_hbm_bandwidth_model->issue_compute(
            node_num_ops, node_tensor_size, wlhd);
    } else {
        if (!node.is_cpu_op) {
            hw_resource->tics_gpu_ops += runtime;
        }
        sys->register_event(this, EventType::General, wlhd, runtime);
    }

    if (execution_mode_ == ExecutionDriven::ExecutionMode::Online &&
        !stats->online_history_preserved()) {
        auto& online_stat = online_statistics_state_or_fail(node.global_id);
        online_stat.compute_utilization = perf / sys->peak_perf;
        // Zero-intensity nodes have no memory utilization; leave the field
        // unset instead of dividing by 0 (a non-finite value fails closed in
        // the compact online aggregation).
        if (operational_intensity != 0.0) {
            online_stat.memory_utilization =
                (perf / operational_intensity) / sys->local_mem_bw;
        }
        if (sys->trace_enabled) {
            workload_logger_
                ->debug("operation_intensity={}, perf={}, elapsed_time={} "
                        "local_mem_latency_ns={} "
                        "compute_utilization={} memory_utilization={} tensor_size={} "
                        "num_ops={}",
                        operational_intensity, perf, elapsed_time,
                        sys->local_mem_latency,
                        online_stat.compute_utilization.value(),
                        online_stat.memory_utilization.value_or(0),
                        tensor_size, num_ops);
        }
    } else {
        // Static ET and history-preserving online microbenchmarks retain the
        // legacy complete per-node Statistics record.
        auto& op_stat = this->stats->get_operator_statistics(node.global_id);
        op_stat.operation_intensity = operational_intensity;
        op_stat.compute_utilization = perf / sys->peak_perf;
        // Same zero-denominator skip as the compact online branch above.
        if (operational_intensity != 0.0) {
            op_stat.memory_utilization =
                (perf / operational_intensity) / sys->local_mem_bw;
        }
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
                        op_stat.memory_utilization.value_or(0), tensor_size,
                        num_ops);
        }
    }
}

void Workload::issue_comm(const ExecutionDriven::NodeView& node) {
    // Fail-closed, not throw: see the Sys::call_events swallowing note in
    // issue_comp.
    if (node.is_cpu_op) {
        workload_logger_->critical("Comm node should not be on CPU");
        exit(EXIT_FAILURE);
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
        workload_logger_->critical("Unknown comm node type");
        exit(EXIT_FAILURE);
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
    // Keep comm_size on the live NodeStore record in compact service mode
    // (terminal communication fact; pinned by
    // statistics_online_compaction_test). No global Statistics map entry is
    // written: OperatorStatistics::comm_size had no reader and was removed
    // with the dead achieved-bandwidth chain.
    if (execution_mode_ == ExecutionDriven::ExecutionMode::Online &&
        !stats->online_history_preserved()) {
        online_statistics_state_or_fail(node.global_id).comm_size = comm_size;
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
                            // chakra runtimes are in microseconds and the
                            // GraphSource adapter already converted them into
                            // nanoseconds
                            runtime);
        fp->set_notifier(this, EventType::CollectiveCommunicationFinished);
    } else {
        workload_logger_->critical("Unsupported collective comm type");
        exit(EXIT_FAILURE);
    }
}

void Workload::issue_send_comm(const ExecutionDriven::NodeView& node) {
    const auto src = node.comm.src;  // adapter default: this rank
    if (src != this->sys->id) {
        workload_logger_->critical(
            "Send node should be issued by the sender");
        exit(EXIT_FAILURE);
    }
    const auto dst = node.comm.dst;
    const auto size = node.comm.bytes;
    // Compact service mode only: retain the size as a terminal
    // communication fact on the live NodeStore record (no bandwidth
    // consumer exists anymore; OperatorStatistics::comm_size is gone).
    if (execution_mode_ == ExecutionDriven::ExecutionMode::Online &&
        !stats->online_history_preserved()) {
        online_statistics_state_or_fail(node.global_id).comm_size = size;
    }
    const auto tag = node.comm.tag;

    // sh_2.0 N-way HBM contention: the p2p sender is a data endpoint of its
    // own HBM (reads comm bytes out of local HBM). hbm-charge=false marks
    // pass-through traffic (NoC<->SerDes relay at an edge rank); multi-hop
    // transit never issues a node on the passed-through rank at all. The
    // node completes as the join of the network packet event and the local
    // HBM job (hbm_endpoint_joins_ latch, see Workload::call).
    WorkloadLayerHandlerData* wlhd = nullptr;
    if (sys->hbm_bandwidth_contention &&
        local_hbm_bandwidth_model != nullptr && size > 0 && node.comm.hbm_charge) {
        wlhd = new WorkloadLayerHandlerData;
        wlhd->sys_id = sys->id;
        wlhd->workload = this;
        wlhd->node_id = node.global_id;
        hbm_endpoint_joins_[node.global_id] = HbmEndpointJoinState{};
        hbm_endpoint_joins_[node.global_id].completion_event =
            EventType::PacketSent;
        local_hbm_bandwidth_model->issue_comm_read(size, wlhd);
    }

    sim_request snd_req;
    snd_req.srcRank = src;
    snd_req.dstRank = dst;
    snd_req.reqType = UINT8;
    SendPacketEventHandlerData* sehd = new SendPacketEventHandlerData;
    sehd->callable = this;
    if (wlhd == nullptr) {
        wlhd = new WorkloadLayerHandlerData;
        wlhd->node_id = node.global_id;
    }
    sehd->wlhd = wlhd;
    sehd->event = EventType::PacketSent;
    sys->front_end_sim_send(0, Sys::dummy_data, size, UINT8, dst, tag, &snd_req,
                            Sys::FrontEndSendRecvType::NATIVE,
                            &Sys::handleEvent, sehd);
}

void Workload::issue_recv_comm(const ExecutionDriven::NodeView& node) {
    const auto src = node.comm.src;
    const auto dst = node.comm.dst;  // adapter default: this rank
    if (dst != this->sys->id) {
        workload_logger_->critical(
            "Recv node should be issued by the receiver");
        exit(EXIT_FAILURE);
    }
    const auto size = node.comm.bytes;
    // Compact service mode only: retain the size as a terminal
    // communication fact on the live NodeStore record (no bandwidth
    // consumer exists anymore; OperatorStatistics::comm_size is gone).
    if (execution_mode_ == ExecutionDriven::ExecutionMode::Online &&
        !stats->online_history_preserved()) {
        online_statistics_state_or_fail(node.global_id).comm_size = size;
    }
    const auto tag = node.comm.tag;

    // sh_2.0 N-way HBM contention: the p2p receiver is a data endpoint of
    // its own HBM (writes comm bytes into local HBM). hbm-charge=false marks
    // pass-through (NoC->SerDes relay at an edge rank) or the remote-load
    // target whose HBM write is already carried by the serially-following
    // restore node (single charge per byte flow).
    WorkloadLayerHandlerData* wlhd = nullptr;
    if (sys->hbm_bandwidth_contention &&
        local_hbm_bandwidth_model != nullptr && size > 0 && node.comm.hbm_charge) {
        wlhd = new WorkloadLayerHandlerData;
        wlhd->sys_id = sys->id;
        wlhd->workload = this;
        wlhd->node_id = node.global_id;
        hbm_endpoint_joins_[node.global_id] = HbmEndpointJoinState{};
        hbm_endpoint_joins_[node.global_id].completion_event =
            EventType::PacketReceived;
        local_hbm_bandwidth_model->issue_comm_write(size, wlhd);
    }

    sim_request rcv_req;
    RecvPacketEventHandlerData* rcehd = new RecvPacketEventHandlerData;
    if (wlhd == nullptr) {
        wlhd = new WorkloadLayerHandlerData;
        wlhd->node_id = node.global_id;
    }
    rcehd->wlhd = wlhd;
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
            // sh_2.0 N-way HBM contention endpoint join: a comm send/recv or
            // pool MEM node carrying a local HBM job completes only when BOTH
            // async sides have called back (network packet / remote-mem FIFO
            // event AND the LocalHbmBandwidthModel job). The first arrival
            // only decrements the latch and returns: the node stays in
            // flight (no release / record_end / finish_node / wlhd delete),
            // and the second arrival runs the terminal handling below
            // exactly once (idempotent per arrival, either order). The early
            // return also skips the trailing static finish check -- safe
            // because this node has not been finish_node'd yet, so
            // static_all_done() is necessarily false here.
            auto join_it = hbm_endpoint_joins_.find(wlhd->node_id);
            EventType hbm_join_event = EventType::General;
            if (join_it != hbm_endpoint_joins_.end()) {
                hbm_join_event = join_it->second.completion_event;
                // Fail-closed double-fire guard (R8-7): the latch must open
                // on exactly one arrival per side. Side attribution is by
                // event type -- a packet event is always the network side
                // of a comm endpoint, and a General arrival on a
                // packet-joined node (completion_event != General) is
                // always its local-HBM job. A pool MEM node's two sides
                // both deliver General and are indistinguishable here, so
                // only its latch count is guarded. A same-side second
                // arrival is a model-layer mechanism violation: fail
                // closed instead of opening the latch early.
                if (event == EventType::PacketSent ||
                    event == EventType::PacketReceived) {
                    if (join_it->second.network_done) {
                        workload_logger_
                            ->critical("duplicate network-side completion "
                                       "for node id={}",
                                       wlhd->node_id);
                        exit(EXIT_FAILURE);
                    }
                    join_it->second.network_done = true;
                } else if (hbm_join_event != EventType::General) {
                    if (join_it->second.hbm_done) {
                        workload_logger_
                            ->critical("duplicate local-HBM job completion "
                                       "for node id={}",
                                       wlhd->node_id);
                        exit(EXIT_FAILURE);
                    }
                    join_it->second.hbm_done = true;
                }
                if (--join_it->second.pending_completions > 0) {
                    return;
                }
                hbm_endpoint_joins_.erase(join_it);
            }
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
                mark_online_terminal_or_fail(wlhd->node_id);
                hw_resource->release(*nv);
        complete_online_statistics(*nv, Sys::boostedTick());
                // R2 anchor fast path: online-only branch (the static else
                // below is untouched and stays unconditional); nv points
                // into the NodeStore record the registration hook set the
                // flags on.
                if (MetricCollector::instance().enabled() &&
                    nv->metric_complete_anchor) {
                    MetricCollector::instance().on_node_complete(
                        sys->id, wlhd->node_id, Sys::boostedTick());
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

            // (The former p2p network-bandwidth accounting block was removed
            // with record_network_bandwidth -- its only consumer was the
            // no-op; the joined-node hbm_join_event plumbing above survives
            // unchanged for the latch semantics.)

            if (this->sys->track_local_mem) {
                this->local_mem_usage_tracker->recordEnd(node,
                                                         Sys::boostedTick());
            }

            // Step 1-3: unconditional node-terminal record (generic wlhd
            // branch; also reached by AnalyticalRemoteMemory completions,
            // which register_event back into Workload::call); step 1-5
            // reverse-index fill in online mode.
            record_node_terminal(execution_mode_, sys->id, wlhd->node_id,
                                 ExecutionDriven::NodeTerminalStatus::Success,
                                 nv);

            graph_source_->finish_node(wlhd->node_id);
            // Static auto-advance only (see collective branch above).
            if (execution_mode_ == ExecutionDriven::ExecutionMode::Static) {
                issue_dep_free_nodes();
            }

            delete wlhd;
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
            (hw_resource->num_in_flight_hbm_dma_ops == 0) &&
            // 远端 MEM 节点在途计数（SerDes 并发化改造方案 §4/阶段2）：
            // remote MEM 节点只在 Workload::call 终结时释放（hbm-access-mode
            // 节点更是要等端口+HBM 两腿 join 开闸），静态 sim-finish 必须等
            // 该计数归零。这是节点占用数，不是端口流数，不参与任何带宽/
            // 并发分母口径。
            (hw_resource->num_in_flight_remote_mem_ops == 0) &&
            // Local-HBM contention: a still-active local-HBM job always
            // belongs to a node that has not completed, so the slot counters
            // above already cover it; the explicit has_active_jobs() check
            // is a belt-and-braces guard -- a drained resolver with pending
            // endpoint jobs must keep waiting for the model's completion
            // callbacks (they re-enter Workload::call and finish the nodes).
            // (num_in_flight_hbm_dma_ops above: sh-family local-HBM
            // KV-tiering DMA occupies dedicated hardware slots.)
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

    // Non-numeric pg_name must not throw: static-path callers run under
    // Sys::call_events, which swallows std::exception -- a throw here would
    // log a misleading "callable removed before call" and leave the issued
    // node occupied forever. Fail closed instead.
    int comm_group_id = 0;
    try {
        size_t parsed_chars = 0;
        comm_group_id = std::stoi(comm_group_name, &parsed_chars);
        if (parsed_chars != comm_group_name.size()) {
            workload_logger_->critical(
                "For rank {} ET node {}, unparsable communicator group "
                "name {} (trailing characters)",
                sys->id, node.global_id, comm_group_name);
            exit(EXIT_FAILURE);
        }
    } catch (const std::exception&) {
        workload_logger_->critical(
            "For rank {} ET node {}, unparsable communicator group name {}",
            sys->id, node.global_id, comm_group_name);
        exit(EXIT_FAILURE);
    }
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
