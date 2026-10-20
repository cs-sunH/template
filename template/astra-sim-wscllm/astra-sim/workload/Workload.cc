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
#include "astra-sim/workload/WorkspaceAdmission.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include <json/json.hpp>

#include <cassert>
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdlib.h>

using namespace std;
using namespace AstraSim;
using json = nlohmann::json;

typedef ChakraProtoMsg::NodeType ChakraNodeType;
typedef ChakraProtoMsg::CollectiveCommType ChakraCollectiveCommType;

namespace {

// Step 1-5: node-terminal record with the already-resolved online NodeView.
// Every terminal path has the view in hand for release / stats, so passing it
// through avoids a second NodeStore hash lookup for every terminal. The view
// is alive for the whole call, so its c_str() pointers remain valid for the
// duration of the observer hook.
void record_node_terminal(
    int rank, uint64_t node_id, ExecutionDriven::NodeTerminalStatus status,
    const ExecutionDriven::NodeView* online_node) {
    ExecutionDriven::CompletionObserver::instance().record_node_terminal(
        rank, node_id,
        (online_node != nullptr && !online_node->request_id.empty())
            ? online_node->request_id.c_str()
            : nullptr,
        (online_node != nullptr && !online_node->stage.empty())
            ? online_node->stage.c_str()
            : nullptr,
        online_node != nullptr ? online_node->generation : 0,
        Sys::boostedTick(), status);
}

// Strict communicator-group id parse (same as astra-sim-face): a non-numeric
// key or a key with trailing characters fails closed with a diagnosable
// message instead of an uncaught std::invalid_argument / silent truncation.
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
    this->graph_source_ = std::move(graph_source);
    // 2.5-1: capture the NodeStore-backed fast path once (null in static
    // mode / for any other source -- by-id virtuals everywhere then).
    this->node_store_source_ =
        this->graph_source_ != nullptr
            ? this->graph_source_->as_node_store_source()
            : nullptr;
    // The offline static path is retired in this repository (the
    // ETFeederGraphSource adapter was removed with it), so a Static-mode
    // Workload would run every static arm on a null GraphSource. Fail closed
    // with an explicit message instead of the bare segfault.
    if (execution_mode != ExecutionDriven::ExecutionMode::Online) {
        workload_logger_->critical(
            "static ET workload mode is retired in this repository; "
            "construct Workload with ExecutionMode::Online and an injected "
            "GraphSource");
        exit(EXIT_FAILURE);
    }

    // Online mode (step-1-2 execution-mode factory): no ETFeeder, no .et
    // file requirement; the dynamic GraphSource was injected at Sys
    // creation (NodeStore-backed implementation lands in step 1-4).  The
    // retired static path (which opened "<et_filename>.<sys_id>.et" through
    // an ETFeeder) is deleted; et_filename stays in the signature only for
    // upstream call-site compatibility.
    // Step 1-4: the GraphSource is the sole dependency-state owner and is
    // injected at Sys creation (NodeStore-backed source, online mode only);
    // the static ETFeeder adapter (ETFeederGraphSource) was retired with
    // the offline static path.
    this->hw_resource = new HardwareResource(sys->id);
    // Multi-user local-HBM bandwidth contention: one fluid model per rank
    // when the flag is on (Sys force-disables it for local_mem_bw <= 0).
    // The model only interacts with this rank's jobs; NoC multi-hop routes
    // never create jobs on intermediate ranks (endpoint-only charging).
    if (sys->hbm_bandwidth_contention) {
        this->local_hbm_bandwidth_model =
            std::make_unique<LocalHbmBandwidthModel>(sys, this);
    }
    this->sys = sys;
    // Die backend: the per-rank adapter owns the live DieExecutions and
    // the Sys-owned plan cache handle. Constructed only under
    // compute-cost-model=die (the adapter fail-closes without the Sys die
    // runtime, which initialize_sys guarantees when the backend is
    // selected).
    if (this->sys->compute_cost_model == ComputeCostModel::Die) {
        this->die_adapter_ = std::make_unique<DieModelAdapter>(this);
    }
    initialize_comm_groups(comm_group_filename);
    this->stats = new Statistics();
    this->stats->configure_online_history_preservation(
        MetricCollector::instance().preserve_online_operator_history());
}

Workload::~Workload() {
    // Die backend teardown FIRST of all: every live DieExecution cancels
    // its pending event and drops its pending HBM die job while all
    // collaborators (the fluid model, the graph source) are still alive
    // (design doc §6.3: no leak, no double completion, no dangling
    // callbacks). Must precede the fluid-model reset below -- the adapter
    // unwind touches cancel_die_jobs_of on the model.
    if (this->die_adapter_ != nullptr) {
        this->die_adapter_->shutdown();
    }

    // Destroy the HBM fluid model FIRST, while every member it can touch is
    // still alive: the model's destructor drains residual jobs through
    // abandon_local_hbm_join_entry(), which erases from hbm_comm_join_.
    // hbm_comm_join_ is declared after local_hbm_bandwidth_model in
    // Workload.hh, so the implicit member destruction order would already
    // have torn the map down before the model's own destructor ran -- that
    // erase would hit a destroyed map (use-after-destruction UB on the
    // abnormal teardown path: a test harness deleting Sys with comm jobs
    // still in flight).
    local_hbm_bandwidth_model.reset();

    comm_groups.clear();

    // Abort-path ownership: DataSets of collectives still in flight are
    // owned by this map (Workload::call deletes each entry on its normal
    // completion). Without this loop, any Workload torn down mid-collective
    // (construction failure upstream, fatal exit paths under test harnesses)
    // leaks every in-flight DataSet.
    for (auto& entry : collective_comm_wrapper_map) {
        delete entry.second;
    }
    collective_comm_wrapper_map.clear();

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
    // communicator group input file is not given. Exact sentinel only (wl-F12
    // fix, aligned with MetricCollector's manifest sentinel): the legacy
    // substring match treated any path merely CONTAINING "empty" as "no
    // input" and silently skipped the load -- a fail-open shape this
    // repository's missing-input fail-closed contract forbids. Callers pass
    // the literal "empty" (Sys fixtures) or a real path.
    if (comm_group_filename == "empty") {
        comm_groups.clear();
        return;
    }

    ifstream inFile;
    json j;
    inFile.open(comm_group_filename);
    if (!inFile) {
        workload_logger_->critical("cannot open comm group file: {}",
                                   comm_group_filename);
        std::exit(EXIT_FAILURE);
    }
    try {
        inFile >> j;
    } catch (const json::exception& e) {
        // Malformed/empty input throws parse_error (nlohmann
        // allow_exceptions path); route it through the unified channel.
        workload_logger_->critical("comm group file is not valid JSON: {} ({})",
                                   comm_group_filename, e.what());
        std::exit(EXIT_FAILURE);
    }
    if (j.is_null() || j.is_discarded()) {
        workload_logger_->critical(
            "comm group file is empty or not valid JSON: {}",
            comm_group_filename);
        std::exit(EXIT_FAILURE);
    }

    for (json::iterator it = j.begin(); it != j.end(); ++it) {
        std::string comm_group_name = it.key();
        int comm_group_id = parse_comm_group_id_strict(comm_group_name);
        // Group id 0 is the default (world) communicator on the graph side
        // ("0" / "" both resolve to nullptr in extract_comm_group), and the
        // CommunicatorGroup constructor rejects ids <= 0 with an exception
        // that would escape this constructor. Fail closed here instead, with
        // the asymmetry spelled out.
        if (comm_group_id <= 0) {
            workload_logger_->critical(
                "communicator group file '{}' defines group '{}' with id {}; "
                "ids must be >= 1 (0 is reserved for the default world "
                "group, which needs no file entry)",
                comm_group_filename, comm_group_name, comm_group_id);
            std::exit(EXIT_FAILURE);
        }

        // Shape and construction failures (wrong JSON type, missing "ranks",
        // empty/duplicate ranks, dimension mismatch) must not escape as
        // exceptions from inside the Workload constructor -- they would
        // terminate the process with no diagnosis. Same fail-closed channel
        // as parse_comm_group_id_strict above.
        try {
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
                    "communicator group '{}': a group must be a rank array "
                    "or an object with ranks and dimensions",
                    comm_group_name);
                std::exit(EXIT_FAILURE);
            }

            comm_groups[comm_group_id] =
                std::make_shared<CommunicatorGroup>(
                    comm_group_id, involved_NPUs, sys, dimension_sizes);
        } catch (const std::exception& e) {
            workload_logger_->critical(
                "malformed communicator group '{}' in {}: {}",
                comm_group_name, comm_group_filename, e.what());
            std::exit(EXIT_FAILURE);
        }
    }
}

void Workload::issue_pytorch_pg_metadata(
    const ExecutionDriven::NodeView& node) {
    // For read comm groups from torch, might overwrite previous.
    std::string pg_info = node.inputs_values;
    if (pg_info.empty()) {
        return;
    }

    // pg metadata is an inner JSON array wrapped in two quote characters on
    // each side; a shorter payload cannot hold one (substr would throw).
    // Fail closed instead of silently skipping the process group.
    if (pg_info.size() < 4) {
        workload_logger_->critical(
            "malformed pg metadata on node {}: inputs_values is {} bytes, "
            "too short for the wrapped pg payload",
            node.global_id, pg_info.size());
        std::exit(EXIT_FAILURE);
    }
    try {
        pg_info = pg_info.substr(2, pg_info.size() - 4);
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
            // pgNameInt + 1 (below) is the group id; at INT32_MAX the
            // increment would be signed-integer overflow (UB) instead of a
            // catchable error.
            if (pgNameInt ==
                std::numeric_limits<int32_t>::max()) {
                workload_logger_->critical(
                    "malformed pg metadata on node {}: pg name '{}' overflows "
                    "the group id space",
                    node.global_id, pgName);
                std::exit(EXIT_FAILURE);
            }
            auto existing = this->comm_groups.find(pgNameInt);
            if (existing != this->comm_groups.end() && existing->second &&
                existing->second->get_id() == pgNameInt + 1 &&
                existing->second->matches_definition(involved_NPUs)) {
                continue;
            }
            // 4.1 fix (2026-09-29): a same-id replacement whose RANK
            // MEMBERSHIP changed is not a supportable live operation. The
            // stream-window base is a pure function of the group id, so
            // after the C1-2 release-first order the new group silently
            // recycles the old group's window while the old group's
            // retained DataSets may still hold in-flight streams on it --
            // the new group's tags alias the old streams (mismatched
            // pairing / hangs). The producer never writes a replacement
            // today (inputs_values is always ""), so the unsupported shape
            // fails closed with a named message instead of silently
            // corrupting the tag space. Same-ranks redefinitions (e.g. a
            // dims-only metadata view difference) stay legal: the registry
            // refcounts the identical lease.
            if (existing != this->comm_groups.end() && existing->second &&
                existing->second->get_id() == pgNameInt + 1 &&
                existing->second->involved_NPUs != involved_NPUs) {
                workload_logger_->critical(
                    "malformed pg metadata on node {}: communicator group "
                    "{} is being redefined with a different membership "
                    "({} -> {} ranks); a live same-id membership change "
                    "would recycle the group's stream window while the old "
                    "group's streams may still be in flight",
                    node.global_id, pgNameInt + 1,
                    existing->second->involved_NPUs.size(),
                    involved_NPUs.size());
                std::exit(EXIT_FAILURE);
            }
            // F1-CPP C1-2 fix (2026-09-29): release the previous group's
            // stream-window lease BEFORE constructing the replacement. The
            // old order (construct first, replace the map entry second)
            // leased the new group's window while the old group still held
            // it, so a legal same-id redefinition whose membership changed
            // hit the DB1 same-id-different-membership panic even though
            // the terminal state has exactly one group. (Same-fingerprint
            // redefinitions never panicked: the registry reference-counts
            // identical leases.) The window above means any constructor
            // failure below is process-fatal anyway (caught -> exit), so
            // the erased-old-group state is never observable.
            this->comm_groups.erase(pgNameInt);
            // To ensure pgName > 0
            auto cg = std::make_shared<CommunicatorGroup>(
                pgNameInt + 1, involved_NPUs, sys);
            this->comm_groups[pgNameInt] = cg;
        }
    } catch (const std::exception& e) {
        workload_logger_->critical("malformed pg metadata on node {}: {}",
                                   node.global_id, e.what());
        std::exit(EXIT_FAILURE);
    }
}

void Workload::issue_dep_free_nodes() {
    // Step 1-4: iterate the GraphSource free views (ascending id, byte-exact
    // order of the pre-phase-1 std::set copy). Post-commit deferred path
    // only: the sole call sites are the online completion hook's per-rank
    // deferred issue passes and the GraphBatchCommitter issue_rank callback
    // (main_online.cc, steps 1-6/1-8/1-11). Workload::call never
    // auto-advances (a bare General event fails closed) and static mode is
    // fail-closed at construction.
    graph_source_->for_each_dep_free([&](const auto& nv) {
        // Availability check on the NodeView. Strategy mode keeps the real
        // serialized physics (path-2 removal 2026-08-18: the replay-only
        // concurrent calibrated-COMP bypass was deleted with the replay
        // route).
        if (!hw_resource->is_available(nv)) {
            return;
        }
        // P2 (2026-09-28): a dispatch failure must name the failing node.
        // The production fatal channels (the commit-period try/catch in
        // main_online.cc and the deferred issue-pass barrier) print the
        // propagated message; without the identity the log only says
        // "MEM_LOAD/MEM_STORE ..." with no pointer at the offending node.
        try {
            const bool coarse_workspace_comp =
                sys->coarse_die_runtime != nullptr &&
                nv.node_type == ChakraNodeType::COMP_NODE &&
                !nv.is_timer_op && !nv.is_cpu_op &&
                nv.compute.op_role != ExecutionDriven::OpRole::ControlMarker;
            if (!coarse_workspace_comp) {
                issue(nv);
                return;
            }

            if (die_adapter_ == nullptr ||
                sys->workspace_admission == nullptr) {
                throw std::runtime_error(
                    "coarse_die_v2 COMP requires a workspace admission "
                    "ledger before issue");
            }
            auto prepared = die_adapter_->prepare_coarse_plan(nv);
            auto admission =
                die_adapter_->try_acquire_workspace(prepared);
            if (admission.status ==
                WorkspaceAdmission::AcquireStatus::Blocked) {
                // Leave this node in the GraphSource free set. A lease
                // release or changed capacity snapshot schedules the next
                // deferred issue pass; no busy retry is created here.
                return;
            }
            if (admission.status ==
                WorkspaceAdmission::AcquireStatus::Impossible) {
                throw std::runtime_error(
                    "coarse_die_v2 Plan workspace " +
                    std::to_string(prepared.workspace_bytes) +
                    " bytes exceeds empty-instance HBM capacity for node " +
                    std::to_string(nv.global_id));
            }
            issue_prepared_coarse(nv, std::move(prepared),
                                  std::move(admission.lease));
        } catch (const std::exception& e) {
            throw std::runtime_error(
                "node issue failed (id=" +
                std::to_string(nv.global_id) +
                " type=" + std::to_string(
                               static_cast<uint64_t>(nv.node_type)) +
                " name=\"" + nv.name + "\"): " + e.what());
        }
    });
}

void Workload::issue(const ExecutionDriven::NodeView& node) {
    issue_impl(node, nullptr, nullptr);
}

void Workload::issue_prepared_coarse(
    const ExecutionDriven::NodeView& node,
    DieModelAdapter::PreparedCoarsePlan prepared,
    WorkspaceAdmission::Lease lease) {
    if (die_adapter_ == nullptr)
        throw std::runtime_error("prepared coarse issue requires its die adapter");
    // Reject a stale or foreign Plan/lease before graph take or GPU occupy.
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
    // state (non-NodeStore source keeps the by-id virtual).
    ExecutionDriven::OnlineRecordRef taken_record;
    if (node_store_source_ != nullptr) {
        taken_record =
            node_store_source_->store().take_node_record(node.global_id);
    } else {
        graph_source_->take_node(node.global_id);
    }
    this->hw_resource->occupy(node);
    // stats->record_end will be called in Workload::call
    start_online_statistics(node, Sys::boostedTick(), taken_record.stats);
    // Side-band metrics observation only; does not touch the node, the
    // dependency resolver, or the event queue (doc sec.5.5).
    // R2 (2026-08-29) anchor fast path: consults the sparse NodeView flag
    // set by the B-1.5 registration hook; unanchored nodes (the
    // overwhelming majority) skip the two-level hash lookup entirely.
    if (MetricCollector::instance().enabled() && node.metric_issue_anchor) {
        MetricCollector::instance().on_node_issue(sys->id, node.global_id,
                                                  Sys::boostedTick());
    }
    // Path-2 removal follow-up (2026-09-26): the sys->replay_only dispatch
    // branch was deleted with the replay-only dispatch route (--online-mode
    // replay; the issue_replay operator issue path below is not that
    // route); the node-type switch below is the sole dispatch.
    if ((node.node_type == ChakraNodeType::MEM_LOAD_NODE) ||
        (node.node_type == ChakraNodeType::MEM_STORE_NODE)) {
        // 2026-09-24 (face A.2 alignment): the remote memory backend was
        // removed; MEM nodes have no issue path and must fail loudly
        // like the old NO_MEMORY_EXPANSION exit(1) did.
        throw std::runtime_error(
            "MEM_LOAD/MEM_STORE nodes require the remote memory "
            "backend, which this repository removes (2026-09-24)");
    } else if (node.node_type == ChakraNodeType::COMP_NODE) {
        if (sys->compute_cost_model == ComputeCostModel::Die) {
            // Die backend: the frozen dispatch order lives in
            // issue_die_comp; the legacy roofline/fluid path below is
            // unreachable under this backend.
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

void Workload::issue_metadata(const ExecutionDriven::NodeView& node) {
    issue_pytorch_pg_metadata(node);
    this->skip_invalid(node);  // for proper dependancy resolving
}

void Workload::issue_replay(const ExecutionDriven::NodeView& node) {
    WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
    wlhd->node_id = node.global_id;
    uint64_t runtime = 1ul;
    if (node.compute.runtime_ns != 0ul) {
        // the online GraphBatch carries runtimes in ns directly (the
        // retired static ET adapter's us->ns conversion no longer exists;
        // the parser reads the value unconverted)
        runtime = node.compute.runtime_ns;
    }
    sys->register_event(this, EventType::General, wlhd, runtime);
}

void Workload::issue_die_comp(
    const ExecutionDriven::NodeView& node,
    DieModelAdapter::PreparedCoarsePlan* const prepared,
    WorkspaceAdmission::Lease* const lease) {
    // Contract §7.3 dispatch order: is_timer_op -> old timer path;
    // is_cpu_op -> issue_replay; op_role=control_marker -> the legacy
    // micro-job semantics verbatim; die_spec -> DieModelAdapter
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
        // contract §2); reaching it here means a producer that bypassed
        // the parser. Fail closed instead of letting a calibrated runtime
        // silently bypass the die execution.
        throw std::runtime_error(
            "die backend requires runtime_ns == 0 on nodes carrying a "
            "die_op_spec (node " +
            std::to_string(node.global_id) + ")");
    }
    if (node.compute.op_role == ExecutionDriven::OpRole::ControlMarker) {
        // Legacy micro-job semantics, byte-for-byte (contract §7.3): with
        // the legacy HBM fluid model the marker drains a (1 op, 1 byte) COMP
        // job (including its local latency phase). Coarse mode uses the same
        // one-byte service as a dedicated canonical endpoint with the same
        // one-shot latency; it does not enter the typed COMP Plan path.
        // Without an HBM facade it remains a 1 ns event. No macro-group
        // launch, workspace lease, typed COMP Plan, or PE/VEC stages are
        // created; the endpoint Runtime owns its ordinary endpoint Plan.
        WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
        wlhd->node_id = node.global_id;
        if (local_hbm_bandwidth_model != nullptr) {
            if (sys->coarse_die_runtime != nullptr) {
                local_hbm_bandwidth_model->issue_control_marker(wlhd);
            } else {
                local_hbm_bandwidth_model->issue_compute(1, 1, wlhd);
            }
        } else {
            // (wscllm note: the write-only tics_gpu_ops accumulator was
            // removed repo-wide on 2026-09-27; the occupancy invariant is
            // the live num_in_flight_gpu_comp_ops counter.)
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
    // Compatibility statistics only (contract §2): num_ops/tensor_size
    // stay visible to the metric side-band; the die execution never
    // charges them as a second cost source.
    if (MetricCollector::instance().enabled()) {
        MetricCollector::instance().on_compute_issue(
            sys->id, node.compute.num_ops, node.compute.tensor_size);
    }
    if (sys->coarse_die_runtime != nullptr) {
        if (prepared == nullptr || lease == nullptr || !lease->active()) {
            throw std::runtime_error(
                "coarse_die_v2 COMP reached issue without its prepared Plan "
                "and pre-issue workspace lease");
        }
        die_adapter_->issue_prepared_coarse_plan(
            node, std::move(*prepared), std::move(*lease));
    } else {
        if (prepared != nullptr || lease != nullptr) {
            throw std::runtime_error(
                "legacy DieModel COMP received a coarse workspace token");
        }
        die_adapter_->issue_die(node);
    }
}

void Workload::issue_comp(const ExecutionDriven::NodeView& node) {
    if (sys->compute_cost_model == ComputeCostModel::Die) {
        // Die-backend red line (design doc §6.4): the legacy closed-form
        // roofline / fluid COMP path must never run under the die
        // compute-cost-model backend -- reaching here means the COMP
        // dispatch order regressed.
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
        throw std::runtime_error(
            "remote operand pipeline loads are not supported in this "
            "repository (no remote memory backend)");
    }

    // Same 1ns floor as issue_replay's default: a sub-nanosecond roofline
    // time (e.g. a zero-FLOP node with no local-mem-latency term) must not
    // schedule a zero-delta event.
    uint64_t runtime =
        std::max<uint64_t>(1ul, static_cast<uint64_t>(elapsed_time * 1e9));  // sec -> ns
    // Step 1-8: online execution-driven calibration. The online GraphBatch
    // (graph_batch_builder.py) carries planner-LUT-aligned durations so the
    // engine timeline is order-isomorphic to the Python decision sequence
    // (fail-closed order consumption). The static .et
    // path (retired in this repo; kept as the historical comparison) never
    // set runtime_ns (duration_micros=0, parsed as 0) and kept the roofline
    // above -- byte-for-byte identical. Same honor pattern as
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
        // Falls through to the shared operator-statistics tail below.
        local_hbm_bandwidth_model->issue_compute(
            node_num_ops, node_tensor_size, wlhd);
    } else {
        sys->register_event(this, EventType::General, wlhd, runtime);
    }

    if (!stats->online_history_preserved()) {
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
                        online_stat.memory_utilization.value_or(0), tensor_size,
                        num_ops);
        }
    } else {
        // Static ET and history-preserving online microbenchmarks retain the
        // legacy complete per-node Statistics record.
        auto& op_stat = this->stats->get_operator_statistics(node.global_id);
        op_stat.compute_utilization = perf / sys->peak_perf;
        // Same zero-denominator skip as the compact online branch above.
        if (operational_intensity != 0.0) {
            op_stat.memory_utilization =
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
                        op_stat.compute_utilization.value(),
                        op_stat.memory_utilization.value_or(0), tensor_size, num_ops);
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
    // involved_dim arrives from the online GraphBatch parse: a bool_list
    // attribute is type-checked at parse time, and an ABSENT attribute
    // leaves the vector empty -- the commit preflight then fails closed
    // (involved_dim must be non-empty and match the declared dimension
    // count). The retired ET adapter's 4x-true default no longer exists.
    const std::vector<bool>& involved_dims = node.coll.involved_dim;

    auto comm_group_owner = extract_comm_group(node);
    CommunicatorGroup* comm_group = comm_group_owner.get();
    const auto comm_type =
        static_cast<ChakraCollectiveCommType>(node.coll.comm_type);
    const auto comm_size = node.coll.bytes;
    const auto comm_priority = node.coll.priority;  // default 0u

    // P2 endpoint charge (4.2 symmetry fix, 2026-09-29): every collective
    // endpoint moves its payload through the local HBM exactly like the
    // p2p send/recv endpoints (read+write merged into one comm_size
    // charge). The original P2 fix covered only ALL_REDUCE, so AG/A2A/RS
    // paid only the byte-blind MemBus fixed delays -- the same systematic
    // underestimate P2 closed, ~20.5us on a 32MiB collective on an
    // exclusive rank. Shared pre-condition: the job is created BEFORE the
    // collective so a synchronously-completing degenerate collective still
    // finds its pending HbmCommJoin entry.
    // Charging basis (2026-10-01 note): the charge uses the APPLICATION
    // comm_size (the declared coll bytes), deliberately NOT the possibly
    // uplifted per-chunk stream size from Sys::determine_chunk_size (which
    // can raise a small collective's wire bytes to splits*total_nodes).
    // The endpoint HBM model mirrors the p2p/COMP convention of charging
    // the producer-declared payload; changing this basis is a timing-model
    // decision, not a bug fix.
    const bool charge_hbm_endpoint =
        local_hbm_bandwidth_model != nullptr && comm_size > 0;
    if (charge_hbm_endpoint) {
        WorkloadLayerHandlerData* hbm_wlhd = new WorkloadLayerHandlerData;
        hbm_wlhd->node_id = node.global_id;
        hbm_comm_join_[node.global_id] = HbmCommJoin{false, false};
        local_hbm_bandwidth_model->issue_comm_read(comm_size, hbm_wlhd);
    }
    if (comm_type == ChakraCollectiveCommType::ALL_REDUCE) {
        // The 2026-09-28 P2 note originally lived HERE, covering only the
        // ALL_REDUCE branch; the 2026-09-29 4.2 symmetry fix moved the
        // charge to the shared pre-dispatch block above (see
        // charge_hbm_endpoint) and this branch-specific copy was left
        // behind, describing the pre-symmetry world. It is folded into the
        // shared block now: all four collective types charge their endpoint
        // bytes identically, complete at the join of the collective and the
        // HBM job (CollectiveCommunicationFinished consults the same
        // HbmCommJoin table the p2p callbacks use), and the job is created
        // BEFORE generate_* so a synchronously-completing degenerate
        // collective still finds the pending join entry.
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
        // Broadcast has no collective implementation here.  The online commit
        // boundary rejects BROADCAST graphs; reaching this branch means an
        // unvalidated input slipped past it (static ET path or a bypassed
        // preflight).  Fail closed with a direct critical + exit (rather
        // than a throw): in online mode Sys::call_events also stops the run
        // on any exception, but only with a generic "callback raised"
        // diagnosis -- exiting here names the collective and the node at
        // the source.  On the retired static arm call_events genuinely
        // swallowed std::exception and continued, which would strand the
        // already-occupied node in flight forever (wl-metrics-E2); the
        // direct exit keeps both arms fail-closed.  Never replay the node
        // as a local event: that would silently drop the communication,
        // network cost, or collective semantics.
        workload_logger_->critical(
            "BROADCAST collective is not implemented; broadcast graphs must "
            "be rejected at the commit boundary (sys.id={}, node={})",
            sys->id, node.global_id);
        exit(EXIT_FAILURE);
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

    sim_request snd_req{};
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

    // Value-initialize every field: the rendezvous path (and any backend)
    // copies this request, so no field may stay indeterminate.
    sim_request rcv_req{};
    rcv_req.srcRank = (uint32_t)src;
    rcv_req.dstRank = (uint32_t)dst;
    rcv_req.reqType = UINT8;
    RecvPacketEventHandlerData* rcehd = new RecvPacketEventHandlerData;
    rcehd->wlhd = new WorkloadLayerHandlerData;
    rcehd->wlhd->node_id = node.global_id;
    rcehd->workload = this;
    rcehd->event = EventType::PacketReceived;
    // The default ctor leaves sys_id 0; the fail-closed "armed with N
    // consumers on sys X" diagnostic in Sys::handleEvent reads it, so a
    // non-zero rank must not be misreported as sys 0 (2026-09-30).
    rcehd->sys_id = sys->id;
    sys->front_end_sim_recv(0, Sys::dummy_data, size, UINT8, src, tag, &rcv_req,
                            Sys::FrontEndSendRecvType::NATIVE,
                            &Sys::handleEvent, rcehd);
}

void Workload::skip_invalid(const ExecutionDriven::NodeView& node) {
    const auto node_id = node.global_id;
    // Claim terminal delivery before any observer/statistics side effect.
    // Duplicate callbacks are a mechanism error rather than an idempotent
    // second aggregate.
    mark_online_terminal_or_fail(node_id);
    // Step 1-3: unconditional node-terminal record (独立于 metrics 开关).
    // Step 1-5: the reverse index is filled from the GraphSource view.
    // Skipped is explicit: the node never executed (INVALID_NODE / metadata
    // / zero tensor_size); whether Skipped satisfies a watch is the watch's
    // own policy, never a default.
    record_node_terminal(sys->id, node_id,
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
    // issue() already occupied the node and started its statistics state;
    // pair them here (release + compact-service completion).
    hw_resource->release(node);
    complete_online_statistics(node, Sys::boostedTick());
    // R2 (2026-08-29) anchor fast path: consults the sparse NodeView flag
    // (set at B-1.5 registration).
    if (MetricCollector::instance().enabled() && node.metric_complete_anchor) {
        MetricCollector::instance().on_node_complete(sys->id, node_id,
                                                     Sys::boostedTick());
    }
}

void Workload::call(EventType event, CallData* data) {
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

        if (hbm_comm_join_.count(node_id) > 0) {
            // P2 (2026-09-28): HBM-joined ALL_REDUCE endpoint (the job was
            // created in issue_coll_comm), same two-event join as p2p: the
            // dataset callback only records the collective's completion;
            // the node completes when the endpoint HBM job has also
            // finished (maybe_complete_hbm_joined_comm below).  The DataSet
            // wrapper is retired here either way.
            delete collective_wrapper;
            hbm_comm_join_[node_id].network_done = true;
            maybe_complete_hbm_joined_comm(node_id);
            return;
        }

        // 2.5-1: one NodeStore find serves the whole terminal sequence
        // (view / terminal claim / mutable stats / dependency release);
        // the handle stays valid for the whole callback (NodeStore GC
        // contract: erase only at the committer's quiescent point).
        const ExecutionDriven::OnlineRecordRef online_record =
            node_store_source_ != nullptr
                ? node_store_source_->store().find_record(node_id)
                : ExecutionDriven::OnlineRecordRef{};
        const ExecutionDriven::NodeView* nv =
            online_record.node != nullptr
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
        // R2 anchor fast path: nv points into the NodeStore record the
        // registration hook set the flags on.
        if (MetricCollector::instance().enabled() &&
            nv->metric_complete_anchor) {
            MetricCollector::instance().on_node_complete(
                sys->id, node_id, Sys::boostedTick());
        }

        // Step 1-3: unconditional node-terminal record (collective branch);
        // step 1-5 reverse-index fill.
        record_node_terminal(sys->id, node_id,
                             ExecutionDriven::NodeTerminalStatus::Success, nv);

        // 2.5-1: dependency release through the handle (no re-find);
        // non-NodeStore source keeps the by-id virtual.
        if (online_record) {
            node_store_source_->store().finish_node_record(online_record);
        } else {
            graph_source_->finish_node(node_id);
        }

        delete collective_wrapper;

    } else {
        if (data == nullptr) {
            // A bare General event used to be the static auto-advance
            // trigger (fire()); the post-commit deferred path calls
            // issue_dep_free_nodes() directly (steps 1-6/1-11). Reaching
            // this is a mechanism violation: fail closed.
            workload_logger_
                ->critical("bare General event reached Workload::call; the "
                           "post-commit path issues dep-free nodes "
                           "directly (step 1-8)");
            exit(EXIT_FAILURE);
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

}

void Workload::finish_generic_node(uint64_t node_id) {
    // Extracted verbatim from the former generic wlhd branch of
    // Workload::call: node release / stats / metrics / terminal record /
    // dependency release.  Shared by the plain register_event completions,
    // the LocalHbmBandwidthModel COMP completions, and the joined p2p comm
    // completions.
    // 2.5-1: one NodeStore find serves the whole terminal sequence
    // (view / terminal claim / mutable stats / dependency release);
    // the handle stays valid for the whole callback (NodeStore GC
    // contract: erase only at the committer's quiescent point).
    const ExecutionDriven::OnlineRecordRef online_record =
        node_store_source_ != nullptr
            ? node_store_source_->store().find_record(node_id)
            : ExecutionDriven::OnlineRecordRef{};
    const ExecutionDriven::NodeView* nv =
        online_record.node != nullptr
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
    // R2 anchor fast path: nv points into the NodeStore record the
    // registration hook set the flags on.
    if (MetricCollector::instance().enabled() && nv->metric_complete_anchor) {
        MetricCollector::instance().on_node_complete(
            sys->id, node_id, Sys::boostedTick());
    }

    // Step 1-3: unconditional node-terminal record (generic wlhd branch;
    // completion paths register_event back into Workload::call);
    // step 1-5 reverse-index fill.
    record_node_terminal(sys->id, node_id,
                         ExecutionDriven::NodeTerminalStatus::Success, nv);

    // 2.5-1: dependency release through the handle (no re-find);
    // non-NodeStore source keeps the by-id virtual.
    if (online_record) {
        node_store_source_->store().finish_node_record(online_record);
    } else {
        graph_source_->finish_node(node_id);
    }
}

void Workload::finish_cancelled_coarse_node(const uint64_t node_id) {
    const ExecutionDriven::NodeView* nv = graph_source_->lookup_ptr(node_id);
    if (nv == nullptr) {
        throw std::runtime_error(
            "cancelled coarse COMP references unknown online node id=" +
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
    record_node_terminal(sys->id, node_id,
                         ExecutionDriven::NodeTerminalStatus::Skipped, nv);
    graph_source_->finish_node(node_id);
}

void Workload::maybe_complete_hbm_joined_comm(uint64_t node_id) {
    auto it = hbm_comm_join_.find(node_id);
    if (it == hbm_comm_join_.end()) {
        return;  // already fired (idempotence guard) or not a joined node
    }
    if (!(it->second.network_done && it->second.hbm_done)) {
        return;  // still waiting for the other side
    }
    // Erase BEFORE completing (defensive ordering): the online mode never
    // re-enters the issue path from a completion (issue passes are driven
    // by main_online), but ANY re-entrant lookup for this node must observe
    // an already-fired join -- erase-first keeps that invariant without
    // depending on the caller's discipline.
    hbm_comm_join_.erase(it);
    finish_generic_node(node_id);
}

void Workload::abandon_local_hbm_join_entry(uint64_t node_id) {
    // Teardown-only (see the header declaration): erase the pending join;
    // no completion is delivered and no metric anchor fires -- the process
    // is already on a fatal/teardown path.
    hbm_comm_join_.erase(node_id);
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
    // OnlineStatisticsState; the legacy OperatorStatistics record
    // (history-preserving microbenchmarks) has no vector field, so only
    // compute/memory cross over there. This repository is online-only, so
    // the compact branch is the production path.
    if (!stats->online_history_preserved()) {
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
