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
#include "astra-sim/workload/MetricCollector.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include <json/json.hpp>

#include <cassert>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
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

// Strict communicator-group key parsing: '7x' previously became 7 and an
// out-of-range id escaped as an uncaught std::out_of_range. Fail loudly with
// the offending key instead (wl-1of1#2).
// Group ids must also be positive (matching the generator, which rejects
// pg_id <= 0): key "0" would build a group no ET node can ever reach
// (extract_comm_group special-cases pg_name "0" as the default all-rank
// group without a table lookup), and a negative id yields a negative
// stream-id band base that folds outside the COLLECTIVE tag window. The
// set_id-side guard was a bare assert compiled out of release builds until
// 2026-10-01, when it was upgraded to a Release-active sys_panic.
int parse_comm_group_id_strict(const std::string& comm_group_name) {
    try {
        size_t pos = 0;
        const int comm_group_id = std::stoi(comm_group_name, &pos);
        if (pos != comm_group_name.size()) {
            LoggerFactory::get_logger("workload")
                ->critical("invalid communicator group key '{}' (trailing "
                           "characters)",
                           comm_group_name);
            exit(EXIT_FAILURE);
        }
        if (comm_group_id <= 0) {
            LoggerFactory::get_logger("workload")
                ->critical("invalid communicator group key '{}' (group ids "
                           "must be positive integers)",
                           comm_group_name);
            exit(EXIT_FAILURE);
        }
        return comm_group_id;
    } catch (const std::exception& e) {
        LoggerFactory::get_logger("workload")
            ->critical("invalid communicator group key '{}': {}",
                       comm_group_name, e.what());
        exit(EXIT_FAILURE);
    }
}

}  // namespace

Workload::Workload(Sys* sys, string et_filename, string comm_group_filename,
                   ExecutionDriven::ExecutionMode execution_mode,
                   std::shared_ptr<ExecutionDriven::GraphSource> graph_source) {
    this->workload_logger_ = LoggerFactory::get_logger("workload");
    this->execution_mode_ = execution_mode;
    this->graph_source_ = std::move(graph_source);
    // 2.5-1: capture the NodeStore-backed fast path once (null in static
    // mode / for any other source -- by-id virtuals everywhere then).
    this->node_store_source_ =
        this->graph_source_ != nullptr
            ? this->graph_source_->as_node_store_source()
            : nullptr;

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
    // Known limitation: the per-rank hardware resource model is never
    // parametrized -- one NPU, giving one in-flight slot per resource class
    // (CPU / GPU compute / GPU comm, see HardwareResource::occupy).
    this->hw_resource = new HardwareResource(sys->id, execution_mode);
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
    // Stage B (die backend): the per-rank adapter owns the live
    // DieExecutions and the Sys-owned plan cache handle. Constructed only
    // under compute-cost-model=die (the adapter fail-closes without the
    // Sys die runtime, which initialize_sys guarantees when the backend is
    // selected).
    if (this->sys->compute_cost_model == ComputeCostModel::Die) {
        this->die_adapter_ = std::make_unique<DieModelAdapter>(this);
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
    // Stage B teardown FIRST: every live DieExecution cancels its pending
    // event and drops its pending HBM die job while all collaborators (the
    // fluid model, the graph source) are still alive (design doc §6.3: no
    // leak, no double completion, no dangling callbacks).
    if (this->die_adapter_ != nullptr) {
        this->die_adapter_->shutdown();
    }

    comm_groups.clear();

    // R1-C3 fix: collective DataSets whose CollectiveCommunicationFinished
    // callback never ran (online-mode early termination, exception unwind)
    // stayed in this map and leaked. The normal path always erases its entry
    // before deleting, so this only sees never-completed wrappers.
    for (auto& kv : collective_comm_wrapper_map) {
        delete kv.second;
    }
    collective_comm_wrapper_map.clear();
    collective_comm_node_id_map.clear();

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
    const ExecutionDriven::NodeView& node, Tick start_time,
    ExecutionDriven::OnlineStatisticsState* pre_resolved) {
    if (stats->online_history_preserved()) {
        stats->record_start(node, start_time);
        return;
    }
    // 2.5-1: pre_resolved is the issue-path handle's stats slot -- part of
    // the same record `node` views (asserted); nullptr = resolve by id as
    // before (skip_invalid and other no-handle callers).
    assert(pre_resolved == nullptr ||
           pre_resolved == &node.online_statistics);
    stats->record_online_service_start(
        node,
        pre_resolved != nullptr
            ? *pre_resolved
            : online_statistics_state_or_fail(node.global_id),
        start_time);
}

void Workload::complete_online_statistics(
    const ExecutionDriven::NodeView& node, Tick end_time,
    ExecutionDriven::OnlineStatisticsState* pre_resolved) {
    if (stats->online_history_preserved()) {
        stats->record_end(node, end_time);
        return;
    }
    // 2.5-1: pre_resolved is the terminal-path handle's stats slot -- part
    // of the same record `node` views (asserted); nullptr = resolve by id.
    assert(pre_resolved == nullptr ||
           pre_resolved == &node.online_statistics);
    stats->complete_online_service_operator(
        node,
        pre_resolved != nullptr
            ? *pre_resolved
            : online_statistics_state_or_fail(node.global_id),
        end_time);
}

void Workload::mark_online_terminal_or_fail(
    uint64_t node_id, const ExecutionDriven::OnlineRecordRef* pre_resolved) {
    if (execution_mode_ != ExecutionDriven::ExecutionMode::Online) {
        return;
    }
    // 2.5-1: pre_resolved (the terminal-path handle) reuses the mint's
    // find; the by-id virtual stays as the exact-behavior fallback.
    if (pre_resolved != nullptr && *pre_resolved) {
        if (!node_store_source_->store().claim_terminal_observed(
                *pre_resolved)) {
            workload_logger_->critical(
                "duplicate, unknown, or unissued online terminal callback "
                "for node {}",
                node_id);
            std::exit(EXIT_FAILURE);
        }
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

void Workload::initialize_comm_groups(string comm_group_filename) {
    // communicator group input file is not given. Exact-equality sentinel
    // (same rule as the Logging.cc log-path sentinels and the cxxopts
    // default "empty"): the old substring match silently zeroed the
    // comm groups and skipped the existence check for ANY path containing
    // "empty" (e.g. a directory named empty_run/).
    if (comm_group_filename == "empty") {
        comm_groups.clear();
        return;
    }

    ifstream inFile;
    json j;
    inFile.open(comm_group_filename);
    // S6 fix (2026-09-28): fail-closed config loading. The bare
    // `inFile >> j` had two failure shapes: an unopenable file failed OPEN
    // (j stayed null, the loop below silently ran zero iterations -- the run
    // proceeded with no comm groups), and a malformed file threw an
    // uncaught nlohmann parse_error (terminate). Both now exit through the
    // repo's standard config-load shape with the file path and the reason.
    if (!inFile.is_open()) {
        workload_logger_->critical("cannot open communicator group file: {}",
                                   comm_group_filename);
        std::exit(EXIT_FAILURE);
    }
    try {
        inFile >> j;
    } catch (const std::exception& e) {
        workload_logger_->critical(
            "failed to parse communicator group file {}: {}",
            comm_group_filename, e.what());
        std::exit(EXIT_FAILURE);
    }

    // R2-3 fix (2026-09-28): gate the top level on objectness before the
    // loop -- it.key() below (and in the catch) throws json::invalid_iterator
    // when iterating a non-object, which used to escape the Workload and Sys
    // constructors as an uncaught exception -> terminate.
    if (!j.is_object()) {
        workload_logger_->critical(
            "communicator group file {} must contain a JSON object mapping "
            "group names to rank arrays or {{ranks, dimensions}} objects",
            comm_group_filename);
        std::exit(EXIT_FAILURE);
    }

    for (json::iterator it = j.begin(); it != j.end(); ++it) {
        // R2-3 fix (2026-09-28): entry-level type errors inside the loop
        // body (a missing "ranks" key -> json::out_of_range, wrong-typed
        // rank/dimension elements -> json::type_error, the explicit shape
        // throw below, and a throwing CommunicatorGroup constructor) used to
        // escape through the Workload and Sys constructors (no try on that
        // chain) -> terminate, without the structured critical line the S6
        // fix introduced for the file-level failures. Route them through the
        // same fail-closed shape. Reading it.key() in the catch is safe: the
        // objectness gate above guarantees an object iterator.
        try {
            std::string comm_group_name = it.key();
            int comm_group_id = parse_comm_group_id_strict(comm_group_name);

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
                    "Communicator group must be a rank array or an object "
                    "with ranks and dimensions");
            }

            comm_groups[comm_group_id] = std::make_shared<CommunicatorGroup>(
                comm_group_id, involved_NPUs, sys, dimension_sizes);
        } catch (const std::exception& e) {
            workload_logger_->critical(
                "invalid communicator group entry '{}' in {}: {}",
                it.key(), comm_group_filename, e.what());
            std::exit(EXIT_FAILURE);
        }
    }
}

void Workload::issue_pytorch_pg_metadata(
    const ExecutionDriven::NodeView& node) {
    // Ordering contract (R2-HN3): within one committed graph batch, every pg
    // metadata node that declares a group is committed before the first
    // collective node that references it -- the builder always emits
    // declaration nodes ahead of their consumers and the committer preserves
    // that order, so extract_comm_group never sees a reference whose
    // declaration is still pending inside the same batch (that shape would
    // fail closed there with "communicator group not found").
    // For read comm groups from torch, might overwrite previous.
    std::string pg_info = node.inputs_values;
    if (pg_info.empty()) {
        return;
    }
    // Fail closed on malformed payloads: the pg metadata is carried as an
    // inner JSON array wrapped in two quote characters on each side, so a
    // shorter inputs_values cannot hold one. substr(2, size - 4) on such
    // input would either throw std::out_of_range (size < 2, uncaught ->
    // terminate) or silently underflow the count (2 <= size < 4); reject
    // it with a structured error instead of crashing or failing open.
    if (pg_info.size() < 4) {
        workload_logger_->critical(
            "malformed pg metadata on node {}: inputs_values is {} bytes, "
            "too short for the wrapped pg payload (expected \"'...'\")",
            node.global_id, pg_info.size());
        std::exit(EXIT_FAILURE);
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

            // Same strict parse as initialize_comm_groups (S-C1): "7x"
            // must not silently collapse to id 7 and displace a
            // file-defined group; non-positive ids are rejected as well.
            const int32_t pgNameInt = parse_comm_group_id_strict(pgName);
            // Same key==id convention as initialize_comm_groups: the map key
            // is the raw pg name (extract_comm_group looks groups up by that
            // name) and the group id equals the key, so the dedup below
            // genuinely recognizes a same-key group whether it was read from
            // the comm-group file or registered by an earlier metadata node.
            // The previous id=pgNameInt+1 offset made the id check always
            // fail for file-initialized groups (id==key), letting metadata
            // silently replace them and discard their cached comm_plans.
            auto existing = this->comm_groups.find(pgNameInt);
            if (existing != this->comm_groups.end() && existing->second &&
                existing->second->get_id() == pgNameInt &&
                existing->second->matches_definition(involved_NPUs)) {
                continue;
            }
            // R9-A1 fix (2026-09-29): a same-key group with a DIFFERENT
            // definition used to be silently replaced here. The dedup
            // predicate compares dimensions too, and the pg metadata payload
            // carries no dimensions, so it always missed a file-defined
            // shaped group (this repo's only shaped-group producer) and the
            // unshaped metadata group displaced it along with its cached
            // comm_plans -- a silent behavior change for every collective on
            // that group. Same key + different definition is a configuration
            // contradiction: fail closed instead of silently displacing.
            if (existing != this->comm_groups.end() && existing->second &&
                existing->second->get_id() == pgNameInt) {
                workload_logger_->critical(
                    "pg metadata on node {} redefines communicator group {} "
                    "with a different definition than the registered one "
                    "(file-defined shaped groups cannot be displaced by "
                    "unshaped pg metadata)",
                    node.global_id, pgNameInt);
                std::exit(EXIT_FAILURE);
            }
            auto cg = std::make_shared<CommunicatorGroup>(
                pgNameInt, involved_NPUs, sys);
            this->comm_groups[pgNameInt] = cg;
        }
    } catch (const std::exception& e) {
        // S6 fix (2026-09-28): the former catch-only-cerr failed OPEN -- a
        // malformed pg metadata payload left the previous/stale group state
        // in place and the run continued. The function's own header comment
        // (and the short-payload guard above) already promise fail-closed;
        // make the parse/processing errors honor it the same way.
        workload_logger_->critical("malformed pg metadata on node {}: {}",
                                   node.global_id, e.what());
        std::exit(EXIT_FAILURE);
    }
}

void Workload::issue_dep_free_nodes() {
    // H1 fix (2026-09-29): a synchronous completion inside issue() (zero-byte
    // collective -> DataSet(0) fires set_notifier inline; HBM harvest inside
    // issue_compute) re-enters this function while the outer
    // graph_source_->for_each_dep_free snapshot iteration is still running.
    // The nested pass used to issue (take_node) siblings that the stale
    // outer snapshot would visit later, and the outer revisit then
    // double-took a node -> runtime_error -> terminate. The guard defers the
    // nested pass to a fresh snapshot taken after the outer pass finishes.
    if (issuing_dep_free_) {
        issue_dep_free_rerun_needed_ = true;
        return;
    }
    issuing_dep_free_ = true;
    do {
        issue_dep_free_rerun_needed_ = false;
        // Step 1-4: iterate the GraphSource free views (ascending id,
        // byte-exact order of the pre-phase-1 std::set copy). Static
        // auto-advance only -- call() gates this behind the execution mode;
        // online mode issues only through the post-commit deferred path
        // (steps 1-6/1-11), which calls this function directly per rank.
        const auto issue_ready_node = [&](const ExecutionDriven::NodeView& node) {
            const bool coarse_workspace_comp =
                sys->coarse_die_runtime != nullptr &&
                node.node_type == ChakraNodeType::COMP_NODE &&
                !node.is_timer_op && !node.is_cpu_op &&
                node.compute.op_role !=
                    ExecutionDriven::OpRole::ControlMarker;
            if (!coarse_workspace_comp) {
                issue(node);
                return;
            }
            if (execution_mode_ != ExecutionDriven::ExecutionMode::Online) {
                throw std::runtime_error(
                    "coarse_die_v2 requires Online bridge v2 workspace "
                    "capacity snapshots");
            }
            if (die_adapter_ == nullptr || sys->workspace_admission == nullptr) {
                throw std::runtime_error(
                    "coarse_die_v2 COMP requires workspace admission before "
                    "issue");
            }
            auto prepared = die_adapter_->prepare_coarse_plan(node);
            auto admission = die_adapter_->try_acquire_workspace(prepared);
            if (admission.status ==
                WorkspaceAdmission::AcquireStatus::Blocked) {
                // Keep the dependency-free candidate unoccupied. A new
                // capacity snapshot or released sibling lease retries it.
                return;
            }
            if (admission.status ==
                WorkspaceAdmission::AcquireStatus::Impossible) {
                throw std::runtime_error(
                    "coarse Plan workspace " +
                    std::to_string(prepared.workspace_bytes) +
                    " bytes exceeds empty-instance HBM capacity for node " +
                    std::to_string(node.global_id));
            }
            issue_prepared_coarse(node, std::move(prepared),
                                  std::move(admission.lease));
        };
        graph_source_->for_each_dep_free([&](const auto& nv) {
            if (execution_mode_ == ExecutionDriven::ExecutionMode::Online) {
                // Step 1-8: online availability check on the NodeView
                // (et_node is nullptr in online mode). Strategy mode keeps the
                // real serialized physics (path-2 removal 2026-08-18: the
                // replay-only concurrent calibrated-COMP bypass was deleted with
                // the replay route).
                if (hw_resource->is_available(nv)) {
                    issue_ready_node(nv);
                }
            } else {
                auto node = graph_source_->et_node(nv.global_id);
                if (hw_resource->is_available(node)) {
                    issue_ready_node(nv);
                }
            }
        });
    } while (issue_dep_free_rerun_needed_);
    issuing_dep_free_ = false;
}

void Workload::issue(const ExecutionDriven::NodeView& node) {
    issue_impl(node, nullptr, nullptr);
}

void Workload::issue_prepared_coarse(
    const ExecutionDriven::NodeView& node,
    DieModelAdapter::PreparedCoarsePlan prepared,
    WorkspaceAdmission::Lease lease) {
    if (die_adapter_ == nullptr) {
        throw std::runtime_error(
            "prepared coarse issue requires its die adapter");
    }
    // This full owner/Plan/task check precedes take_node() and occupy().
    die_adapter_->validate_prepared_coarse_plan(node, prepared, lease);
    issue_impl(node, &prepared, &lease);
}

void Workload::issue_impl(
    const ExecutionDriven::NodeView& node,
    DieModelAdapter::PreparedCoarsePlan* const prepared,
    WorkspaceAdmission::Lease* const lease) {
    auto logger = workload_logger_;
    if (sys->trace_enabled) {
        logger->debug("issue,sys->id={}, tick={}, node->id={}, "
                      "node->name={}, node->type={}",
                      sys->id, Sys::boostedTick(), node.global_id, node.name,
                      static_cast<uint64_t>(node.node_type));
    }

    // 2.5-1: one find serves mark_issued AND the mutable statistics
    // state; static mode / non-NodeStore source keeps the by-id virtual.
    ExecutionDriven::OnlineRecordRef taken_record;
    if (node_store_source_ != nullptr) {
        taken_record =
            node_store_source_->store().take_node_record(node.global_id);
    } else {
        graph_source_->take_node(node.global_id);
    }
    // Static path: the ETFeederNode handle for the node-bound consumers
    // (HardwareResource / Statistics / local_mem tracker) is fetched through
    // the GraphSource. Step 1-8: the online path (et_node == nullptr) uses
    // the NodeView overloads.
    std::shared_ptr<Chakra::FeederV3::ETFeederNode> et_node = nullptr;
    if (execution_mode_ == ExecutionDriven::ExecutionMode::Online) {
        this->hw_resource->occupy(node);
        // stats->record_end will be called in Workload::call
        start_online_statistics(node, Sys::boostedTick(),
                                taken_record.stats);
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
            // A.2 (2026-09-05): the remote memory backend was removed; MEM
            // nodes have no issue path and must fail loudly like the old
            // NO_MEMORY_EXPANSION exit(1) did.
            throw std::runtime_error(
                "MEM_LOAD/MEM_STORE nodes require the remote memory "
                "backend, which this repository removes (A.2, 2026-09-05)");
        } else if (node.node_type == ChakraNodeType::COMP_NODE) {
            if (sys->compute_cost_model == ComputeCostModel::Die) {
                // Die backend (stage B, contract §7.3): the frozen dispatch
                // order lives in issue_die_comp; the legacy roofline/fluid
                // path below is unreachable under this backend.
                issue_die_comp(node, prepared, lease);
            } else if (!this->sys->roofline_enabled) {
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
    // Known limitation: the ET format carries no marker identifying a
    // METADATA_NODE as a PyTorch process-group record, so every metadata
    // node is speculatively handed to issue_pytorch_pg_metadata (which
    // tolerates that and returns early on empty inputs_values).
    issue_pytorch_pg_metadata(node);
    this->skip_invalid(node);  // for proper dependancy resolving
}

void Workload::issue_replay(const ExecutionDriven::NodeView& node) {
    WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
    wlhd->node_id = node.global_id;
    uint64_t runtime = 1ul;
    if (node.compute.runtime_ns != 0ul) {
        // runtime_ns is always consumed in nanoseconds: the GraphSource
        // adapter converts chakra micros at parse time (view_of), and the
        // online GraphBatch carries planner durations directly in ns
        runtime = node.compute.runtime_ns;
    }
    if (!node.is_cpu_op) {
        hw_resource->tics_gpu_ops += runtime;
    }
    sys->register_event(this, EventType::General, wlhd, runtime);
}

void Workload::issue_die_comp(
    const ExecutionDriven::NodeView& node,
    DieModelAdapter::PreparedCoarsePlan* const prepared,
    WorkspaceAdmission::Lease* const lease) {
    // Contract §7.3 dispatch order (frozen 2026-10-03): is_timer_op -> old
    // timer path; is_cpu_op -> issue_replay; op_role=control_marker -> the
    // legacy micro-job semantics verbatim; die_spec -> DieModelAdapter
    // (runtime_ns must be 0); anything else -> fail-closed. The legacy
    // roofline closed form is NEVER reachable under the die backend.
    if (node.is_timer_op) {
        issue_replay(node);
        return;
    }
    if (node.is_cpu_op) {
        issue_replay(node);
        return;
    }
    if ((node.compute.die_spec != nullptr ||
         node.compute.die_task_json != nullptr) &&
        node.compute.runtime_ns != 0) {
        // The GraphBatch parser already rejects this combination (N12 /
        // contract §2); reaching it here means a producer that bypassed the
        // parser (e.g. a static ET view). Fail closed instead of letting a
        // calibrated runtime silently bypass the die execution.
        throw std::runtime_error(
            "die backend requires runtime_ns == 0 on nodes carrying a "
            "die_op_spec (node " +
            std::to_string(node.global_id) + ")");
    }
    if (node.compute.op_role == ExecutionDriven::OpRole::ControlMarker) {
        // Legacy micro-job semantics, byte-for-byte (contract §7.3): with
        // the legacy HBM model the marker drains its original (1 op, 1 byte)
        // COMP job; coarse_die_v2 bridges the one physical byte through its
        // canonical DIE endpoint with the same configured local-memory
        // delay. Without an HBM facade it remains a 1 ns event. No typed
        // COMP Plan, PE/VEC stages, or extra graph launch is created.
        WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
        wlhd->node_id = node.global_id;
        if (local_hbm_bandwidth_model != nullptr) {
            if (sys->coarse_die_runtime != nullptr) {
                local_hbm_bandwidth_model->issue_control_marker(wlhd);
            } else {
                local_hbm_bandwidth_model->issue_compute(1, 1, wlhd);
            }
        } else {
            hw_resource->tics_gpu_ops += 1;
            sys->register_event(this, EventType::General, wlhd, 1);
        }
        return;
    }
    if (node.compute.die_spec == nullptr &&
        (sys->coarse_die_runtime == nullptr ||
         node.compute.die_task_json == nullptr)) {
        // Contract §2/§7.3: a real COMP without a die_op_spec is a
        // generator defect -- never a silent roofline/replay fallback.
        throw std::runtime_error(
            "die backend requires die_op_spec on real COMP node " +
            std::to_string(node.global_id) +
            " (op_role=compute without a spec; refusing to fall back to "
            "roofline or replay)");
    }
    // Compatibility statistics only (contract §2): num_ops/tensor_size stay
    // visible to the metric side-band; the die execution never charges them
    // as a second cost source.
    if (MetricCollector::instance().enabled()) {
        MetricCollector::instance().on_compute_issue(
            sys->id, node.compute.num_ops, node.compute.tensor_size);
    }
    if (sys->coarse_die_runtime != nullptr) {
        if (prepared == nullptr || lease == nullptr || !lease->active()) {
            throw std::runtime_error(
                "coarse_die_v2 COMP reached issue without its prepared Plan "
                "and live workspace lease");
        }
        die_adapter_->issue_prepared_coarse_plan(
            node, std::move(*prepared), std::move(*lease));
    } else {
        if (prepared != nullptr || lease != nullptr) {
            throw std::runtime_error(
                "legacy die COMP unexpectedly carries coarse admission state");
        }
        die_adapter_->issue_die(node);
    }
}

void Workload::issue_comp(const ExecutionDriven::NodeView& node) {
    if (sys->compute_cost_model == ComputeCostModel::Die) {
        // Stage B red line (design doc §6.4): the die backend must not run
        // the legacy closed-form roofline / fluid COMP path -- reaching here
        // means the COMP dispatch order regressed.
        throw std::runtime_error(
            "Workload::issue_comp (legacy roofline path) must never run "
            "under the die compute-cost-model backend");
    }

    if (!this->sys->roofline_enabled) {
        throw std::runtime_error(
            "Roofline model is not enabled for non-replay comp");
    }

    if (node.is_cpu_op) {
        throw std::runtime_error("Roofline is only available for GPU nodes");
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

    const uint64_t node_num_ops = node.compute.num_ops;
    const uint64_t node_tensor_size = node.compute.tensor_size;
    double num_ops = static_cast<double>(node_num_ops);
    double tensor_size = static_cast<double>(node_tensor_size);

    // if tensor_size is 0 during roofline mode, this is an invalid node
    if (tensor_size == 0) {
        skip_invalid(node);
        return;
    }

    // (2026-09-30, R2 audit) Reject the unsupported remote-operand load
    // BEFORE allocating wlhd: the old order leaked the handler on this
    // throw path (process terminates anyway, but the leak-free discipline
    // costs nothing when the check simply moves up).
    if (node.compute.has_remote_weight_bytes) {
        throw std::runtime_error(
            "remote operand pipeline loads are not supported in this "
            "repository (no remote memory backend)");
    }

    WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
    wlhd->node_id = node.global_id;

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

    // R5-Q5 fix: elapsed_time comes from user data divided by configured
    // rates; a pathological configuration (tiny but positive peak_perf) can
    // push elapsed_time * 1e9 outside the uint64 range, where the floating
    // -> integer conversion is UB. Anything >= 1e18 ns (~31.7 simulated
    // years) cannot be a real run; fail closed instead.
    const double runtime_ns = elapsed_time * 1e9;
    if (!std::isfinite(runtime_ns) || runtime_ns >= 1e18) {
        workload_logger_->critical(
            "computed comp runtime {} ns is not representable (elapsed_time "
            "{} s); check peak-perf/local-mem-bw and the node's num_ops",
            runtime_ns, elapsed_time);
        exit(EXIT_FAILURE);
    }
    uint64_t runtime = static_cast<uint64_t>(runtime_ns);  // sec -> ns
    // Step 1-8: online execution-driven calibration. The online GraphBatch
    // (graph_batch_builder.py) carries planner-LUT-aligned durations so the
    // engine timeline is order-isomorphic to the Python decision sequence
    // (fail-closed order consumption). Static .et nodes get runtime_ns
    // through the same view_of micros->ns conversion (traces without
    // duration_micros parse it as 0), so they keep the roofline above --
    // byte-for-byte preserved. Same honor pattern as issue_replay's
    // runtime_ns branch.
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
        local_hbm_bandwidth_model->issue_compute(
            node_num_ops, node_tensor_size, wlhd);
    } else {
        // (2026-09-30) is_cpu_op threw at function entry, so this arm is
        // always a GPU op -- the old conditional was dead.
        hw_resource->tics_gpu_ops += runtime;
        // A zero runtime (roofline elapsed < 1ns and no calibrated
        // runtime_ns) would register a same-tick event and break the event
        // queue's strict-increase invariant; degrade to 1ns (exedrv-1of3#4).
        sys->register_event(this, EventType::General, wlhd,
                            std::max<Tick>(runtime, 1));
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
    // Known limitation: the ET collective comm_tag (used upstream to tell
    // apart two collectives of the same type inside one process group) is
    // not parsed into CollAttrs and is ignored on this path -- only
    // comm_priority participates in collective scheduling here. p2p
    // send/recv do honor their comm.tag.
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
        // Known limitation: BROADCAST has no collective implementation in
        // this repository -- it is modeled as a rank-local timing-only
        // replay: a single-stream DataSet completes after the node's
        // recorded runtime in nanoseconds (default 1 ns) without any
        // network or communicator interaction.
        uint64_t runtime = 1ul;
        if (node.compute.runtime_ns != 0ul) {
            // Same nanosecond contract as issue_replay: the adapter converts
            // chakra micros at parse time (NodeStore view_of, BROADCAST case)
            runtime = node.compute.runtime_ns;
        }
        DataSet* fp = new DataSet(1);
        fp->retain_communicator_group(comm_group_owner);
        collective_comm_node_id_map[fp->my_id] = node.global_id;
        collective_comm_wrapper_map[fp->my_id] = fp;
        // runtime is already in nanoseconds (the GraphSource adapter
        // converted the chakra micros at parse time).
        sys->register_event(fp, EventType::General, nullptr, runtime);
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
        hbm_wlhd->node_id = node.global_id;
        hbm_comm_join_[node.global_id] = HbmCommJoin{false, false};
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
        hbm_wlhd->node_id = node.global_id;
        hbm_comm_join_[node.global_id] = HbmCommJoin{false, false};
        local_hbm_bandwidth_model->issue_comm_write(size, hbm_wlhd);
    }

    sim_request rcv_req;
    // (2026-10-01) mirror the send side's explicit fields: the analytical
    // backends never dereference the request, but the rendezvous protocol
    // path copies the whole struct (reading srcRank/dstRank) and an
    // uninitialized pass-by-value would leak garbage into that copy.
    rcv_req.srcRank = src;
    rcv_req.dstRank = dst;
    rcv_req.reqType = UINT8;
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
    // (2026-09-30, R2 audit) finish_node may release CHILD nodes into the
    // free set, and this run sits inside the for_each_dep_free value
    // snapshot: without the rerun flag a pass consisting solely of skipped
    // nodes (a pure metadata chain) leaves the children in the free set with
    // no event ever scheduling another pass -- a static-mode stall. Online
    // mode is unaffected in practice (the online protocol carries no
    // metadata/INVALID nodes, and the deferred issue pass re-drains anyway).
    issue_dep_free_rerun_needed_ = true;
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
        // 2.5-1: one NodeStore find serves the whole terminal sequence
        // (view / terminal claim / mutable stats / dependency release);
        // the handle stays valid for the whole callback (NodeStore GC
        // contract: erase only at the committer's quiescent point).
        ExecutionDriven::OnlineRecordRef online_record;
        shared_ptr<Chakra::FeederV3::ETFeederNode> node = nullptr;
        if (execution_mode_ == ExecutionDriven::ExecutionMode::Online) {
            online_record =
                node_store_source_ != nullptr
                    ? node_store_source_->store().find_record(node_id)
                    : ExecutionDriven::OnlineRecordRef{};
            nv = online_record.node != nullptr
                     ? online_record.node
                     : graph_source_->lookup_ptr(node_id);
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
            mark_online_terminal_or_fail(node_id, &online_record);
            hw_resource->release(*nv);
            complete_online_statistics(*nv, Sys::boostedTick(),
                                       online_record.stats);
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

        // 2.5-1: dependency release through the handle (no re-find);
        // static mode / non-NodeStore source keeps the by-id virtual.
        if (online_record) {
            node_store_source_->store().finish_node_record(online_record);
        } else {
            graph_source_->finish_node(node_id);
        }
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
                finish_generic_node(wlhd->node_id);
                delete wlhd;
            }
        }
    }

    // Final end authority (M3-1): shared by every completion route --
    // collective / bare-General completions finalize here, while generic and
    // LocalHbmBandwidthModel-delivered completions finalize inside
    // finish_generic_node (the is_finished guard makes this second check a
    // no-op on those paths).
    maybe_finalize_static_run();
}

void Workload::finish_generic_node(uint64_t node_id) {
    // Extracted verbatim from the former generic wlhd branch of
    // Workload::call: node release / stats / metrics / terminal record /
    // dependency release / static-mode auto-advance.  Shared by the plain
    // register_event completions, the LocalHbmBandwidthModel COMP
    // completions, and the joined p2p comm completions.
    const ExecutionDriven::NodeView* nv = nullptr;
    // 2.5-1: one NodeStore find serves the whole terminal sequence
    // (view / terminal claim / mutable stats / dependency release);
    // the handle stays valid for the whole callback (NodeStore GC
    // contract: erase only at the committer's quiescent point).
    ExecutionDriven::OnlineRecordRef online_record;
    shared_ptr<Chakra::FeederV3::ETFeederNode> node = nullptr;
    if (execution_mode_ == ExecutionDriven::ExecutionMode::Online) {
        online_record =
            node_store_source_ != nullptr
                ? node_store_source_->store().find_record(node_id)
                : ExecutionDriven::OnlineRecordRef{};
        nv = online_record.node != nullptr
                 ? online_record.node
                 : graph_source_->lookup_ptr(node_id);
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
        mark_online_terminal_or_fail(node_id, &online_record);
        hw_resource->release(*nv);
        complete_online_statistics(*nv, Sys::boostedTick(),
                                   online_record.stats);
        // R2 anchor fast path: online-only branch (the static else below is
        // untouched and stays unconditional); nv points into the NodeStore
        // record the registration hook set the flags on.
        if (MetricCollector::instance().enabled() &&
            nv->metric_complete_anchor) {
            MetricCollector::instance().on_node_complete(
                sys->id, node_id, Sys::boostedTick());
        }
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

    if (this->sys->track_local_mem) {
        this->local_mem_usage_tracker->recordEnd(node, Sys::boostedTick());
    }

    // Step 1-3: unconditional node-terminal record (generic wlhd branch;
    // completion paths register_event back into Workload::call);
    // step 1-5 reverse-index fill
    // in online mode.
    record_node_terminal(execution_mode_, sys->id, node_id,
                         ExecutionDriven::NodeTerminalStatus::Success, nv);

    // 2.5-1: dependency release through the handle (no re-find);
    // static mode / non-NodeStore source keeps the by-id virtual.
    if (online_record) {
        node_store_source_->store().finish_node_record(online_record);
    } else {
        graph_source_->finish_node(node_id);
    }
    // Static auto-advance only (see the collective branch in Workload::call).
    if (execution_mode_ == ExecutionDriven::ExecutionMode::Static) {
        issue_dep_free_nodes();
    }
    // M3-1: HBM-delivered completions (the LocalHbmBandwidthModel COMP
    // callback and the HBM side of a joined p2p comm) finish the node here
    // without re-entering Workload::call, so the static run-final check must
    // also run on this path -- otherwise a run whose last completion is
    // HBM-delivered would never report()/notify finished. Runs whose last
    // completion goes through Workload::call already finalize in its tail;
    // the is_finished guard keeps exactly one finalization.
    maybe_finalize_static_run();
}

void Workload::finish_cancelled_coarse_node(const uint64_t node_id) {
    if (execution_mode_ != ExecutionDriven::ExecutionMode::Online) {
        throw std::runtime_error(
            "live coarse COMP cancellation requires an Online graph source");
    }
    const ExecutionDriven::NodeView* nv = graph_source_->lookup_ptr(node_id);
    if (nv == nullptr || nv->node_type != ChakraNodeType::COMP_NODE) {
        throw std::runtime_error(
            "cancelled coarse COMP references an unknown/non-COMP node id=" +
            std::to_string(node_id));
    }
    if (die_cancelled_comp_count_ == std::numeric_limits<uint64_t>::max()) {
        throw std::overflow_error("die_cancelled_comp_count overflow");
    }
    mark_online_terminal_or_fail(node_id);
    ++die_cancelled_comp_count_;
    hw_resource->release(*nv);
    complete_online_statistics(*nv, Sys::boostedTick());
    if (MetricCollector::instance().enabled() && nv->metric_complete_anchor) {
        MetricCollector::instance().on_node_complete(sys->id, node_id,
                                                     Sys::boostedTick());
    }
    record_node_terminal(execution_mode_, sys->id, node_id,
                         ExecutionDriven::NodeTerminalStatus::Skipped, nv);
    graph_source_->finish_node(node_id);
}

void Workload::maybe_finalize_static_run() {
    // Final end authority (方案 §4 步骤 1-2 操作 3): in online mode this
    // side effect is disabled -- a temporarily empty graph must never end the
    // online simulation and drop later injections. The end authority belongs
    // exclusively to the ServiceCoordinator (总体方案 §5.5).
    if (execution_mode_ == ExecutionDriven::ExecutionMode::Online ||
        is_finished) {
        return;
    }
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
        // callbacks (they finish the nodes directly through
        // finish_generic_node, which re-runs this same check).
        (local_hbm_bandwidth_model == nullptr ||
         !local_hbm_bandwidth_model->has_active_jobs()) &&
        // Stage B (die backend), mirror of the M3-1 note above: a
        // DieExecution completes its node through finish_generic_node
        // WITHOUT re-entering Workload::call, and its internal HBM jobs
        // complete through the DieExecution sink (never through the
        // Workload), so a still-executing outer COMP is only visible
        // here through the adapter's live-execution set. A drained
        // resolver must keep waiting for the machine's Done (design doc
        // §6.3 lifecycle red line: is_finished/drain checks must include
        // not-yet-finished DieExecutions).
        (die_adapter_ == nullptr ||
         !die_adapter_->has_active_die_executions())) {
        report();
        sys->comm_NI->sim_notify_finished();
        is_finished = true;
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
    // Erase BEFORE completing: the completion re-enters the issue path
    // (static auto-advance) and any re-entrant lookup for this node must
    // observe an already-fired join.
    hbm_comm_join_.erase(it);
    finish_generic_node(node_id);
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
    finish_generic_node(node_id);
    delete wlhd;
}

void Workload::record_die_utilizations(const uint64_t node_id,
                                       const double compute_utilization,
                                       const double memory_utilization,
                                       const double vector_utilization) {
    // Contract §7.6: the die-only vector_utilization lives on the compact
    // OnlineStatisticsState; the legacy OperatorStatistics record (static ET
    // and history-preserving microbenchmarks) has no vector field, so only
    // compute/memory cross over there.
    if (execution_mode_ == ExecutionDriven::ExecutionMode::Online &&
        !stats->online_history_preserved()) {
        ExecutionDriven::OnlineStatisticsState& state =
            online_statistics_state_or_fail(node_id);
        state.compute_utilization = compute_utilization;
        state.memory_utilization = memory_utilization;
        state.vector_utilization = vector_utilization;
        return;
    }
    auto& op_stat = this->stats->get_operator_statistics(node_id);
    op_stat.compute_utilization = compute_utilization;
    op_stat.memory_utilization = memory_utilization;
}

void Workload::report() {
    // Compact online service has no legacy per-node Statistics history. Check
    // before emitting any report output so this path cannot report zeros.
    stats->ensure_legacy_post_processing_supported();
    Tick curr_tick = Sys::boostedTick();
    // Under replay_only every node type is replayed through issue_replay, so
    // GPU COMM nodes also accumulate tics_gpu_ops while COMP and COMM occupy
    // independent single-slot resources and can overlap in time; the sum can
    // then exceed the wall clock and the unsigned subtraction below would
    // wrap to ~1.8e19. Clamp the exposed-communication figure at zero; when
    // tics_gpu_ops <= curr_tick the printed value is unchanged.
    const Tick gpu_ops_ticks = hw_resource->tics_gpu_ops;
    workload_logger_
        ->info("sys[{}] finished, {} cycles, exposed communication {} cycles.",
               sys->id, curr_tick,
               gpu_ops_ticks < curr_tick ? curr_tick - gpu_ops_ticks : 0);
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

    int comm_group_id = parse_comm_group_id_strict(comm_group_name);
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
