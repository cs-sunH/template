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
#include "astra-sim/workload/FailClosed.hh"
#include "astra-sim/workload/LocalHbmBandwidthModel.hh"
#include "astra-sim/workload/MetricCollector.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "astra-sim/workload/execution_driven/PipelineDescriptor.hh"
#include "astra-sim/workload/execution_driven/RemoteReadExecutor.hh"
#include <json/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdlib.h>
#include <typeinfo>
#include <unistd.h>

using namespace std;
using namespace AstraSim;
using namespace Chakra::FeederV3;
using json = nlohmann::json;

typedef ChakraProtoMsg::NodeType ChakraNodeType;
typedef ChakraProtoMsg::CollectiveCommType ChakraCollectiveCommType;

namespace {

// Strict communicator-group key parsing (ported from the face repo S-C1
// hardening, 2026-09-30): "7x" previously became 7 via std::stoi and an
// out-of-range id escaped as an uncaught std::out_of_range. Fail loudly
// with the offending key instead. Group ids must also be positive: key "0"
// folds onto the default all-rank group semantics and a negative id yields
// a negative stream-id band base outside the COLLECTIVE tag window.
int parse_comm_group_id_strict(const std::string& comm_group_name) {
    try {
        size_t pos = 0;
        const int comm_group_id = std::stoi(comm_group_name, &pos);
        if (pos != comm_group_name.size()) {
            fail_closed("invalid communicator group key '" +
                        comm_group_name + "' (trailing characters)");
        }
        if (comm_group_id <= 0) {
            fail_closed("invalid communicator group key '" +
                        comm_group_name +
                        "' (group ids must be positive integers)");
        }
        return comm_group_id;
    } catch (const std::exception& e) {
        fail_closed("invalid communicator group key '" + comm_group_name +
                    "': " + e.what());
    }
}

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
        // Same fail-closed construction guard as the comm-group file parse
        // in initialize_comm_groups: this runs in the Workload constructor
        // ("new Workload" inside Sys, no try/catch up the chain), so a
        // workload file that opens but fails to parse (corrupt/truncated
        // .et body, duplicate node ids, failed sanity check) would
        // otherwise surface as a bare std::terminate at startup.
        try {
            this->et_feeder = new ETFeeder(workload_filename);
        } catch (const std::exception& e) {
            workload_logger_->critical(
                "failed to construct the ET feeder for workload file {}: {}",
                workload_filename, e.what());
            exit(EXIT_FAILURE);
        }
    }
    // Step 1-4: the GraphSource is the sole dependency-state owner. Static
    // mode wraps the ETFeeder + DependancyResolver in the adapter
    // (ETFeederGraphSource) -- byte-for-byte the pre-phase-1 resolver calls;
    // online mode received an injected NodeStore-backed source at Sys
    // creation and must never fall back to an ETFeeder.
    if (this->graph_source_ == nullptr &&
        execution_mode == ExecutionDriven::ExecutionMode::Static) {
        // Same fail-closed construction guard (see the ETFeeder new above):
        // an adapter-construction throw must not escape this constructor as
        // an uncaught exception (bare std::terminate, no diagnostic).
        try {
            this->graph_source_ =
                std::make_shared<ExecutionDriven::ETFeederGraphSource>(
                    this->et_feeder, sys->id);
        } catch (const std::exception& e) {
            workload_logger_->critical(
                "failed to construct the static graph source for rank {}: {}",
                sys->id, e.what());
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
    // The LocalHbmBandwidthModel exists when either flag asks for it:
    // hbm-bandwidth-contention (new; N-way equal split owning COMP, restore,
    // comm and pool endpoints) or the legacy hbm-kv-restore-bandwidth-sharing
    // (two-user 50/50 semantics preserved for A/B). Inside the model the
    // equal-split math is one N-way implementation; with contention off no
    // comm/pool jobs are ever created, so the observable behavior is exactly
    // the legacy one.
    if (sys->hbm_kv_restore_bandwidth_sharing || sys->hbm_bandwidth_contention) {
        this->local_hbm_bandwidth_model =
            std::make_unique<LocalHbmBandwidthModel>(sys, this);
    }
    // Die backend (joint rollout stage 1): the per-rank adapter owns the
    // live DieExecutions and the Sys-owned plan cache handle. Constructed
    // only under compute-cost-model=die (the adapter fail-closes without
    // the Sys die runtime, which initialize_sys guarantees when the backend
    // is selected).
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
    // Die backend teardown FIRST (design doc §6.3): every live DieExecution
    // cancels its pending event and drops its pending HBM die job while all
    // collaborators (the fluid model, the graph source) are still alive (no
    // leak, no double completion, no dangling callbacks).
    if (this->die_adapter_ != nullptr) {
        this->die_adapter_->shutdown();
    }

    comm_groups.clear();

    // R1-C3 fix (ported 2026-09-30): collective DataSets whose
    // CollectiveCommunicationFinished callback never ran (online-mode early
    // termination, exception unwind) stayed in this map and leaked. The
    // normal path always erases its entry before deleting, so this only
    // sees never-completed wrappers.
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

void Workload::initialize_comm_groups(string comm_group_filename) {
    // communicator group input file is not given. Exact-equality sentinel
    // (ported from the face repo S6-class fix, 2026-09-30): the old
    // substring match silently zeroed the comm groups and skipped the
    // existence check for ANY path containing "empty" (e.g. a directory
    // named empty_run/).
    if (comm_group_filename == "empty") {
        comm_groups.clear();
        return;
    }

    ifstream inFile;
    json j;
    inFile.open(comm_group_filename);
    // Fail-closed config loading (ported from the face repo S6/R2-3 fixes,
    // 2026-09-30): an unopenable file used to leave j null and silently run
    // with no comm groups, a malformed file threw an uncaught parse_error
    // (terminate), a non-object document made every contains() false, and
    // per-entry type errors escaped the Workload/Sys constructors. All four
    // shapes now exit through fail_closed with the file path and the reason.
    if (!inFile.is_open()) {
        fail_closed("cannot open communicator group file: " +
                    comm_group_filename);
    }
    try {
        inFile >> j;
    } catch (const std::exception& e) {
        fail_closed("failed to parse communicator group file " +
                    comm_group_filename + ": " + e.what());
    }
    if (!j.is_object()) {
        fail_closed("communicator group file " + comm_group_filename +
                    " must contain a JSON object mapping group names to "
                    "rank arrays or {ranks, dimensions} objects");
    }

    for (json::iterator it = j.begin(); it != j.end(); ++it) {
        try {
            std::string comm_group_name = it.key();
            int comm_group_id =
                parse_comm_group_id_strict(comm_group_name);

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

            comm_groups[comm_group_id] =
                std::make_shared<CommunicatorGroup>(
                    comm_group_id, involved_NPUs, sys, dimension_sizes);
        } catch (const std::exception& e) {
            fail_closed("invalid communicator group entry '" + it.key() +
                        "' in " + comm_group_filename + ": " + e.what());
        }
    }
}

void Workload::issue_pytorch_pg_metadata(
    const ExecutionDriven::NodeView& node) {
    // For read comm groups from torch, might overwrite previous.
    // Hardening ported from the face repo (2026-09-30): fail-closed on
    // malformed payloads (the former bare substr threw or underflowed on
    // short inputs and the catch-only-cerr failed OPEN, keeping stale group
    // state), strict group-key parsing, and a fail-closed conflict rule for
    // same-key different-definition redefinitions (R9-A1: the unshaped
    // metadata group used to silently displace a file-defined shaped group
    // along with its cached comm_plans). This repo keeps its historical
    // id = key + 1 metadata-group convention unchanged.
    std::string pg_info = node.inputs_values;
    if (pg_info.empty()) {
        return;
    }
    if (pg_info.size() < 4) {
        fail_closed("malformed pg metadata on node " +
                    std::to_string(node.global_id) +
                    ": inputs_values is too short for the wrapped pg "
                    "payload (expected the two quote-wrapped JSON "
                    "characters)");
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

            const int32_t pgNameInt = parse_comm_group_id_strict(pgName);
            auto existing = this->comm_groups.find(pgNameInt);
            if (existing != this->comm_groups.end() && existing->second &&
                (existing->second->get_id() == pgNameInt + 1 ||
                 existing->second->get_id() == pgNameInt) &&
                existing->second->matches_definition(involved_NPUs)) {
                continue;
            }
            // Same key but a different definition is a configuration
            // contradiction: fail closed instead of silently displacing the
            // registered group.
            if (existing != this->comm_groups.end() && existing->second) {
                fail_closed("pg metadata on node " +
                            std::to_string(node.global_id) +
                            " redefines communicator group " + pgName +
                            " with a different definition than the "
                            "registered one");
            }
            // To ensure pgName > 0
            auto cg = std::make_shared<CommunicatorGroup>(
                pgNameInt + 1, involved_NPUs, sys);
            this->comm_groups[pgNameInt] = cg;
        }
    } catch (const std::exception& e) {
        fail_closed("malformed pg metadata on node " +
                    std::to_string(node.global_id) + ": " + e.what());
    }
}

void Workload::issue_dep_free_nodes() {
    // H1/R1-C1 fix (ported 2026-09-30): a synchronous completion inside
    // issue() (zero-byte collective -> DataSet(0) fires set_notifier inline;
    // HBM harvest inside the model's issue path) re-enters this function
    // while the outer ready-batch issue iteration is still running. The nested pass used to issue (take_node) siblings that
    // the stale outer snapshot would visit later, and the outer revisit then
    // double-took a node -> runtime_error -> terminate. The guard defers the
    // next ready enumeration until the outer batch finishes.
    if (issuing_dep_free_) {
        issue_dep_free_rerun_needed_ = true;
        return;
    }
    issuing_dep_free_ = true;
    do {
        issue_dep_free_rerun_needed_ = false;
        using SteadyClock = std::chrono::steady_clock;
        const auto snapshot_start = SteadyClock::now();
        // Capture a ready batch completely before issuing any member. The
        // NodeStore's ordered set already provides incremental
        // membership/deduplication; this stable, ascending id snapshot keeps
        // issue callbacks from injecting a newly-ready child into the current
        // batch. The ET fallback obtains the same vector that its default
        // for_each_dep_free implementation previously traversed.
        // F (dynamic ordered current physical anchor candidate set,
        // 2026-10-11): the online id snapshot is now the ORDINARY
        // (non-anchor-role) free set -- anchor-role ids are enumerated from
        // the executor's current-anchor set inside the merge loop instead of
        // being copied here. Derived/non-NodeStore sources keep their
        // existing default: the full id snapshot through the same call.
        ExecutionDriven::NodeStoreGraphSource* node_store_source = nullptr;
        if (execution_mode_ == ExecutionDriven::ExecutionMode::Online &&
            graph_source_ != nullptr &&
            typeid(*graph_source_) ==
                typeid(ExecutionDriven::NodeStoreGraphSource)) {
            node_store_source =
                static_cast<ExecutionDriven::NodeStoreGraphSource*>(
                    graph_source_.get());
        }
        const uint64_t ready_version =
            node_store_source == nullptr
                ? 0
                : node_store_source->store().free_id_version();
        const auto cached_snapshot_source =
            node_store_source == nullptr
                ? std::shared_ptr<ExecutionDriven::GraphSource>()
                : dep_free_issue_snapshot_source_.lock();
        const bool can_reuse_id_snapshot =
            node_store_source != nullptr &&
            node_store_source->store().snapshot_reuse_eligible() &&
            dep_free_issue_snapshot_valid_ &&
            ready_version != std::numeric_limits<uint64_t>::max() &&
            dep_free_issue_snapshot_version_ == ready_version &&
            cached_snapshot_source != nullptr &&
            cached_snapshot_source.get() == graph_source_.get();
        bool has_id_snapshot = false;
        if (can_reuse_id_snapshot) {
            has_id_snapshot = true;
            ++dep_free_issue_timing_.snapshot_reuse_hits;
        } else {
            // Invalidate before calling the source so a false return or an
            // exception cannot leave a stale vector tagged as current.
            dep_free_issue_snapshot_valid_ = false;
            dep_free_issue_snapshot_source_.reset();
            // Only the EXACT NodeStore source takes the ordinary partition
            // fill (its anchors are enumerated from the executor's
            // current-anchor set). A derived source that overrides the full
            // fill (false/throwing/flipping contract fixtures) keeps the
            // original full-snapshot virtual call, so its overrides still
            // engage unchanged; non-NodeStore sources are on the same
            // compatibility path as before.
            has_id_snapshot =
                node_store_source != nullptr
                    ? graph_source_->fill_dep_free_ordinary_id_snapshot(
                          dep_free_issue_snapshot_ids_)
                    : graph_source_->fill_dep_free_id_snapshot(
                          dep_free_issue_snapshot_ids_);
            if (has_id_snapshot) {
                ++dep_free_issue_timing_.snapshot_full_fills;
                if (node_store_source != nullptr) {
                    const uint64_t filled_version =
                        node_store_source->store().free_id_version();
                    dep_free_issue_snapshot_version_ = filled_version;
                    if (filled_version !=
                            std::numeric_limits<uint64_t>::max() &&
                        node_store_source->store()
                            .snapshot_reuse_eligible()) {
                        dep_free_issue_snapshot_source_ = graph_source_;
                        dep_free_issue_snapshot_valid_ = true;
                    }
                }
            }
        }
        if (!has_id_snapshot) {
            dep_free_issue_snapshot_valid_ = false;
            dep_free_issue_snapshot_source_.reset();
            dep_free_issue_snapshot_views_ = graph_source_->dep_free_nodes();
        }
        // A successfully captured exact-NodeStore ID vector is also proof
        // that each captured ID belonged to the authoritative ready set at
        // this generation. Reuse that proof only during this pass and only
        // while the same source remains eligible and unchanged. The strong
        // owner keeps the snapshot's source alive if a synchronous callback
        // replaces graph_source_ while this pass is still walking the vector.
        std::shared_ptr<ExecutionDriven::GraphSource>
            membership_snapshot_owner;
        uint64_t membership_snapshot_version =
            std::numeric_limits<uint64_t>::max();
        bool membership_snapshot_proof_active = false;
        if (has_id_snapshot && node_store_source != nullptr &&
            node_store_source->store().snapshot_reuse_eligible()) {
            const uint64_t snapshot_version =
                dep_free_issue_snapshot_version_;
            if (snapshot_version != std::numeric_limits<uint64_t>::max()) {
                membership_snapshot_owner = graph_source_;
                membership_snapshot_version = snapshot_version;
                membership_snapshot_proof_active = true;
            }
        }
        std::vector<ExecutionDriven::RemoteReadExecutor::ReadyEndpoint>
            virtual_ready_snapshot;
        if (execution_mode_ == ExecutionDriven::ExecutionMode::Online &&
            remote_read_executor_ != nullptr) {
            virtual_ready_snapshot =
                remote_read_executor_->ready_virtual_endpoints(sys->id);
        }
        const auto snapshot_end = SteadyClock::now();
        dep_free_issue_timing_.snapshot_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                snapshot_end - snapshot_start)
                .count());
        ++dep_free_issue_timing_.snapshot_passes;
        dep_free_issue_timing_.candidates +=
            has_id_snapshot ? dep_free_issue_snapshot_ids_.size()
                            : dep_free_issue_snapshot_views_.size();
        dep_free_issue_timing_.candidates += virtual_ready_snapshot.size();
        // F (2026-10-11): visited current-anchor candidates are added to
        // `candidates` after the walk (they are no longer pre-counted from
        // a ready-set snapshot; future anchors never appear at all).
        uint64_t anchor_visits_this_pass = 0;

        const auto issue_start = SteadyClock::now();
        auto issue_if_available = [&](const auto& nv) {
            if (execution_mode_ == ExecutionDriven::ExecutionMode::Online) {
                // Step 1-8: online availability check on the NodeView
                // (et_node is nullptr in online mode). Strategy mode keeps the
                // real serialized physics (path-2 removal 2026-08-18 deleted the
                // replay-only concurrent calibrated-COMP bypass).
                if (hw_resource->is_available(nv)) {
                    issue(nv);
                    return true;
                }
            } else {
                auto node = graph_source_->et_node(nv.global_id);
                if (hw_resource->is_available(node)) {
                    issue(nv);
                    return true;
                }
            }
            return false;
        };
        auto issue_online_candidate_if_available =
            [&](const ExecutionDriven::NodeView& candidate,
                const ExecutionDriven::NodeView& resource_view) {
                if (!hw_resource->is_available(resource_view)) {
                    return false;
                }
                // The availability gate is read-only. Materialize the
                // candidate only after it passes, but before issue() can take
                // the node or synchronously mutate/re-hash NodeStore.
                const ExecutionDriven::NodeView node = candidate;
                ++dep_free_issue_timing_.node_view_materializations;
                issue(node);
                return true;
            };
        auto is_current_snapshot_member = [&](const uint64_t node_id) {
            if (membership_snapshot_proof_active) {
                ExecutionDriven::GraphSource* const current_source =
                    graph_source_.get();
                if (current_source != nullptr &&
                    current_source == membership_snapshot_owner.get()) {
                    // The owner was admitted only after the exact dynamic
                    // NodeStoreGraphSource check above; pointer identity
                    // preserves that type without another RTTI query here.
                    auto* const current_node_store_source =
                        static_cast<ExecutionDriven::NodeStoreGraphSource*>(
                            current_source);
                    const auto& store = current_node_store_source->store();
                    if (store.snapshot_reuse_eligible() &&
                        store.free_id_version() ==
                            membership_snapshot_version) {
                        // No callback or mutable operation can run between
                        // this test and lookup_ptr. A prior issue or virtual
                        // dispatch that changed the source/set is detected on
                        // the next candidate and keeps this proof disabled.
                        return true;
                    }
                }
                membership_snapshot_proof_active = false;
            }
            return graph_source_->is_dep_free_member(node_id);
        };
        uint64_t issue_calls = 0;
        if (has_id_snapshot) {
            // F (2026-10-11) three-way merge: ordinary physical vector /
            // dynamically fetched current anchors / virtual endpoints, all
            // ascending by store_id. The anchor stream is NOT materialized:
            // every iteration asks the executor for the smallest current
            // anchor id past the walk's last visited id (refreshed at the
            // TOP of each iteration, i.e. AFTER the previous step's issue
            // callbacks), so anchors whose run cursor has not reached them
            // are never enumerated (the old full-snapshot path visited and
            // coordinate-rejected them). A chain A->B in one run still
            // works in-round: issue(A) synchronously advances the run
            // cursor (erase A / insert B into the executor's current set),
            // and the next refresh returns B because B > last visited.
            size_t physical_index = 0;
            size_t virtual_index = 0;
            // The walk's last visited id, updated for EVERY processed
            // candidate (ordinary / virtual / anchor, issued or not). The
            // anchor stream resumes from it, so an anchor that becomes
            // current only after the walk already moved past its id is not
            // revisited in this pass -- it waits for the rerun/deferred
            // pass, exactly like the old snapshot walk's no-revisit rule.
            uint64_t walk_last_visited = 0;
            uint64_t anchor_next = 0;
            bool anchor_next_valid = false;
            const bool anchor_stream_active =
                execution_mode_ == ExecutionDriven::ExecutionMode::Online &&
                remote_read_executor_ != nullptr;
            const auto refresh_anchor_next = [&]() {
                anchor_next_valid =
                    anchor_stream_active &&
                    remote_read_executor_->next_current_anchor_id(
                        static_cast<uint64_t>(sys->id), walk_last_visited,
                        &anchor_next);
            };
            while (true) {
                refresh_anchor_next();
                const uint64_t physical_next =
                    physical_index < dep_free_issue_snapshot_ids_.size()
                        ? dep_free_issue_snapshot_ids_[physical_index]
                        : std::numeric_limits<uint64_t>::max();
                const uint64_t virtual_next =
                    virtual_index < virtual_ready_snapshot.size()
                        ? virtual_ready_snapshot[virtual_index].store_id
                        : std::numeric_limits<uint64_t>::max();
                const uint64_t anchor_candidate =
                    anchor_next_valid ? anchor_next
                                      : std::numeric_limits<uint64_t>::max();
                if (physical_next == std::numeric_limits<uint64_t>::max() &&
                    virtual_next == std::numeric_limits<uint64_t>::max() &&
                    !anchor_next_valid) {
                    break;
                }
                if (anchor_candidate < physical_next &&
                    anchor_candidate < virtual_next) {
                    // Current-anchor candidate (gate 1: the executor's set
                    // already proves the run cursor points at it). Advance
                    // the walk FIRST so the id is never revisited even if
                    // the gates below reject it.
                    walk_last_visited = anchor_candidate;
                    ++anchor_visits_this_pass;
                    // Gate 2: current ready-set membership (the id may have
                    // been taken/finished by an earlier candidate's
                    // synchronous callback). The exact-NodeStore path uses
                    // its ordered set; a compatibility source keeps the old
                    // generic membership recheck.
                    const bool anchor_still_free =
                        node_store_source != nullptr
                            ? node_store_source->store().is_free_id(
                                  anchor_candidate)
                            : graph_source_->is_dep_free_member(
                                  anchor_candidate);
                    if (!anchor_still_free) {
                        continue;
                    }
                    // Gate 3: fixed pass membership -- ids that entered the
                    // free set after this pass's snapshot fill belong to the
                    // next pass (the same rule the ordinary vector gets
                    // from the snapshot itself). The fill drained the
                    // delta, and it stays empty on the version-unchanged
                    // snapshot-reuse path.
                    if (node_store_source != nullptr) {
                        const auto& delta =
                            node_store_source->store()
                                .free_delta_since_snapshot();
                        if (std::binary_search(delta.begin(), delta.end(),
                                               anchor_candidate)) {
                            continue;
                        }
                    }
                    const auto* const current = graph_source_->lookup_ptr(
                        anchor_candidate);
                    if (current == nullptr) {
                        continue;
                    }
                    if (current->compute.exec_role ==
                        ExecutionDriven::ComputeExecRole::
                            RemoteReadPipelineAnchor) {
                        ExecutionDriven::NodeView resource_view;
                        // Anchor binding, dependency, protocol and slot
                        // validation only read executor/NodeStore state and
                        // construct a separate resource view; they do not
                        // issue the anchor or run a completion callback.
                        // This full preflight (including the current-id
                        // recheck inside) runs for every visited candidate
                        // exactly as before.
                        if (!remote_read_executor_->resource_view_for_anchor(
                                *current, &resource_view)) {
                            continue;
                        }
                        if (issue_online_candidate_if_available(*current,
                                                                resource_view)) {
                            ++issue_calls;
                        }
                        continue;
                    }
                    // A non-anchor node registered as a physical endpoint
                    // binding cannot happen (registration validates the
                    // anchor role); fail closed rather than issuing it as
                    // an ordinary node outside the snapshot contract.
                    fail_closed(
                        "current-anchor candidate lacks its anchor exec role");
                }
                const bool issue_virtual = virtual_next < physical_next;
                if (issue_virtual) {
                    const auto& candidate =
                        virtual_ready_snapshot[virtual_index++];
                    walk_last_visited = candidate.store_id;
                    if (hw_resource->is_available(candidate.resource_node) &&
                        remote_read_executor_->dispatch_virtual_endpoint(
                            sys->id, candidate.store_id)) {
                        ++issue_calls;
                    }
                    continue;
                }

                const uint64_t node_id =
                    dep_free_issue_snapshot_ids_[physical_index++];
                walk_last_visited = node_id;
                if (!is_current_snapshot_member(node_id)) {
                    continue;
                }
                const auto* const current = graph_source_->lookup_ptr(node_id);
                if (current == nullptr) {
                    continue;
                }
                if (execution_mode_ ==
                        ExecutionDriven::ExecutionMode::Online &&
                    current->compute.exec_role ==
                        ExecutionDriven::ComputeExecRole::
                            RemoteReadPipelineAnchor) {
                    // Compatibility path: a non-NodeStore source's default
                    // ordinary fill is the FULL id snapshot, so anchors can
                    // still arrive through the vector. Keep the original
                    // anchor gate for them (an online store source filters
                    // anchors out of the vector and never reaches this).
                    ExecutionDriven::NodeView resource_view;
                    if (remote_read_executor_ == nullptr ||
                        !remote_read_executor_->resource_view_for_anchor(
                            *current, &resource_view)) {
                        continue;
                    }
                    if (issue_online_candidate_if_available(*current,
                                                            resource_view)) {
                        ++issue_calls;
                    }
                } else if (execution_mode_ ==
                           ExecutionDriven::ExecutionMode::Online) {
                    if (issue_online_candidate_if_available(*current,
                                                            *current)) {
                        ++issue_calls;
                    }
                } else {
                    // Keep the static id-snapshot path's existing value
                    // lifetime and resource-check order unchanged.
                    const ExecutionDriven::NodeView node = *current;
                    ++dep_free_issue_timing_.node_view_materializations;
                    if (issue_if_available(node)) {
                        ++issue_calls;
                    }
                }
            }
        } else {
            for (const auto& node : dep_free_issue_snapshot_views_) {
                if (issue_if_available(node)) {
                    ++issue_calls;
                }
            }
        }
        const auto issue_end = SteadyClock::now();
        dep_free_issue_timing_.issue_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                issue_end - issue_start)
                .count());
        dep_free_issue_timing_.candidates += anchor_visits_this_pass;
        dep_free_issue_timing_.anchor_candidate_visits += anchor_visits_this_pass;
        dep_free_issue_timing_.issue_calls += issue_calls;
    } while (issue_dep_free_rerun_needed_);
    issuing_dep_free_ = false;
}

void Workload::set_remote_read_executor(
    ExecutionDriven::RemoteReadExecutor* executor) {
    if (execution_mode_ != ExecutionDriven::ExecutionMode::Online ||
        executor == nullptr || remote_read_executor_ != nullptr) {
        fail_closed("invalid online RemoteReadExecutor attachment");
    }
    remote_read_executor_ = executor;
}

void Workload::set_pipeline_kernel_contract(
    std::shared_ptr<const ExecutionDriven::PipelineKernelContract> contract) {
    if (contract == nullptr) {
        fail_closed("cannot install an empty run-frozen pipeline contract");
    }
    contract->validate();
    if (pipeline_kernel_contract_ != nullptr &&
        *pipeline_kernel_contract_ != *contract) {
        fail_closed("Workload pipeline contract changed after first commit");
    }
    pipeline_kernel_contract_ = std::move(contract);
}

std::shared_ptr<const ExecutionDriven::PipelineKernelContract>
Workload::pipeline_kernel_contract() const {
    return pipeline_kernel_contract_;
}

void Workload::publish_ledger_transition(
    const char* kind, const ExecutionDriven::NodeView& node) const {
    if (execution_mode_ != ExecutionDriven::ExecutionMode::Online ||
        ledger_transition_observer_ == nullptr || kind == nullptr ||
        node.compute.exec_role ==
            ExecutionDriven::ComputeExecRole::RemoteReadPipelineAnchor) {
        return;
    }
    ledger_transition_observer_(kind, node);
}

void Workload::issue(const ExecutionDriven::NodeView& node) {
    const bool pipeline_anchor =
        execution_mode_ == ExecutionDriven::ExecutionMode::Online &&
        node.compute.exec_role ==
            ExecutionDriven::ComputeExecRole::RemoteReadPipelineAnchor;
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
        // A physical GraphBatch anchor is a zero-cost structural placeholder.
        // Its mapped COMM/COMP operation occupies the ordinary hardware gate
        // in TileTransportAdapter::submit after the combined-ready check.
        // Occupying the placeholder here would charge a second resource slot.
        if (!pipeline_anchor) {
            this->hw_resource->occupy(node);
        }
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
    if (this->sys->track_local_mem && !pipeline_anchor) {
        this->local_mem_usage_tracker->recordStart(et_node,
                                                   Sys::boostedTick());
    }
    // The v2 ledger records the ordinary logical node at its actual issue
    // point. Structural endpoint anchors are proxies; their charged work is
    // published by RemoteReadExecutor when the mapped endpoint is issued.
    publish_ledger_transition("issue", node);
    if (pipeline_anchor) {
        if (remote_read_executor_ == nullptr ||
            !remote_read_executor_->dispatch_anchor(node)) {
            fail_closed("physical pipeline anchor reached Workload without a "
                        "ready executor endpoint");
        }
        return;
    }
    if (sys->replay_only) {
        issue_replay(node);
    } else {
        if ((node.node_type == ChakraNodeType::MEM_LOAD_NODE) ||
            (node.node_type == ChakraNodeType::MEM_STORE_NODE)) {
            // sh_3.0 fourth-slot dispatch (baseline sh_3.0 Workload
            // issue's MEM routing): MEM_LOAD carrying
            // is_local_hbm_kv_restore routes to the HBM restore model;
            // other MEM nodes go to the remote-memory FIFO.
            if (node.mem.is_local_hbm_kv_restore) {
                issue_local_hbm_kv_restore(node);
            } else {
                issue_remote_mem(node);
            }
        } else if (node.node_type == ChakraNodeType::COMP_NODE) {
            if (sys->compute_cost_model == ComputeCostModel::Die) {
                // Die backend (joint rollout stage 1, contract §7.3): the
                // frozen dispatch order lives in issue_die_comp; the legacy
                // roofline/fluid path below is unreachable under this
                // backend.
                issue_die_comp(node);
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
    // CPU ops are excluded from the GPU busy-tick accumulator (the old
    // sibling tics_cpu_ops counter was write-only and was removed).
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
    // Path-2 removal (2026-08-18): the replay-only instant (1ns) remote MEM
    // completion branch was deleted with the replay route; strategy/static
    // keep the real remote port-FIFO physics.
    // N-way HBM contention pool endpoint: a MEM node annotated
    // hbm-access-mode 1/2 additionally charges this rank's HBM
    // (POOL_READ/POOL_WRITE, bytes = tensor_size); node completion becomes
    // the join of the port transaction and the HBM job. Unannotated MEM
    // nodes (including every NoC/SerDes pass-through) keep the single-sided
    // completion exactly as before.
    if (hbm_endpoint_charge_active() &&
        node.mem.hbm_access_mode != 0 && node.mem.tensor_size > 0) {
        // Fail-closed attr-domain check (contract: 0/absent = none, 1 =
        // local HBM read, 2 = local HBM write). Anything else used to fall
        // through the "!= 1" below into a WRITE, and a value that lost
        // width on the producer side (uint64 -> int truncation) could even
        // arrive negative and silently flip the endpoint leg. Static-path
        // callers run under Sys::call_events, which swallows
        // std::exception, so breach = critical + exit (same convention as
        // issue_comp).
        if (node.mem.hbm_access_mode < 0 ||
            (node.mem.hbm_access_mode != 1 &&
             node.mem.hbm_access_mode != 2)) {
            workload_logger_->critical(
                "invalid hbm-access-mode {} on MEM node {} (rank {}); "
                "legal values are 0 (none), 1 (local HBM read), "
                "2 (local HBM write)",
                node.mem.hbm_access_mode, node.global_id, sys->id);
            exit(EXIT_FAILURE);
        }
        WorkloadLayerHandlerData* hbm_wlhd =
            begin_hbm_join(node.global_id, EventType::General);
        if (node.mem.hbm_access_mode == 1) {
            local_hbm_bandwidth_model->issue_pool_read(
                node.mem.tensor_size, hbm_wlhd);
        } else {
            local_hbm_bandwidth_model->issue_pool_write(
                node.mem.tensor_size, hbm_wlhd);
        }
    }
    sys->remote_mem->issue(node.mem.tensor_size, wlhd);
}

void Workload::issue_local_hbm_kv_restore(
    const ExecutionDriven::NodeView& node) {
    WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
    wlhd->sys_id = sys->id;
    wlhd->workload = this;
    wlhd->node_id = node.global_id;

    const uint64_t tensor_size = node.mem.tensor_size;
    // Side-band restore byte accounting (doc sec.5.2/9.4); the runtime model
    // counters in LocalHbmBandwidthModel track the same bytes as served.
    if (MetricCollector::instance().enabled()) {
        MetricCollector::instance().on_local_hbm_restore_issue(sys->id,
                                                               tensor_size);
    }
    // Path-2 removal (2026-08-18): the replay-only instant (1ns) HBM restore
    // completion branch was deleted with the replay route; strategy/static
    // keep the real LocalHbmBandwidthModel DMA physics.
    if (local_hbm_bandwidth_model != nullptr) {
        local_hbm_bandwidth_model->issue_restore(tensor_size, wlhd);
        return;
    }

    // Fallback DMA path divides by the configured local-mem-bw below. That
    // key parses contains-only with a 0 default, and local_mem_bw <= 0 only
    // auto-disables contention in the Sys config parse -- nothing validates
    // this division. A non-positive bus would produce inf here and an UB
    // uint64 conversion (a garbage, effectively hung node); fail closed
    // instead of silently mis-timing the node.
    if (!std::isfinite(sys->local_mem_bw) || sys->local_mem_bw <= 0) {
        fail_closed(
            "local HBM KV-restore fallback requires a positive finite local-mem-bw "
            "(configure local-mem-bw or enable hbm-bandwidth-contention / "
            "hbm-kv-restore-bandwidth-sharing): sys.id=" +
            std::to_string(sys->id) +
            " node_id=" + std::to_string(node.global_id) +
            " tensor_size=" + std::to_string(tensor_size));
    }

    const double elapsed_seconds =
        (static_cast<double>(sys->local_mem_latency) / 1e9) +
        static_cast<double>(tensor_size) / sys->local_mem_bw;
    const double runtime_ns = elapsed_seconds * 1e9;
    if (!std::isfinite(runtime_ns) || runtime_ns >= 1e18) {
        fail_closed(
            "local HBM KV-restore fallback runtime is not representable: " +
            std::to_string(runtime_ns) + " ns");
    }
    const uint64_t runtime = std::max<uint64_t>(
        1, static_cast<uint64_t>(std::ceil(runtime_ns)));
    sys->register_event(this, EventType::General, wlhd, runtime);
}

void Workload::issue_pipeline_control_marker(
    const ExecutionDriven::NodeView& node,
    WorkloadLayerHandlerData* completion_cookie) {
    if (execution_mode_ != ExecutionDriven::ExecutionMode::Online ||
        completion_cookie == nullptr ||
        completion_cookie->pipeline_sink == nullptr ||
        node.rank != sys->id || node.kind != ExecutionDriven::NodeKind::Compute ||
        node.compute.op_role != ExecutionDriven::OpRole::ControlMarker ||
        node.compute.num_ops != 1 || node.compute.tensor_size != 1) {
        delete completion_cookie;
        fail_closed("invalid virtual 1-op/1-byte pipeline control marker");
    }
    completion_cookie->sys_id = sys->id;
    completion_cookie->workload = this;
    completion_cookie->node_id = node.global_id;

    const bool die_backend =
        sys->compute_cost_model == ComputeCostModel::Die;
    const bool replay_branch = sys->replay_only ||
        (die_backend && (node.is_timer_op || node.is_cpu_op)) ||
        (!die_backend && (!sys->roofline_enabled || node.is_cpu_op));
    if (replay_branch) {
        const uint64_t runtime = node.compute.runtime_ns != 0
            ? node.compute.runtime_ns : uint64_t{1};
        if (!node.is_cpu_op) hw_resource->tics_gpu_ops += runtime;
        sys->register_event(this, EventType::General, completion_cookie, runtime);
        return;
    }

    uint64_t roofline_runtime = 1;
    if (!die_backend) {
        // issue_comp evaluates/validates the roofline even when the HBM
        // contention model later owns the actual 1-op/1-byte duration.
        // Preserve that dispatch contract before choosing the completion
        // backend below.
        if (sys->roofline == nullptr) {
            delete completion_cookie;
            fail_closed("roofline control marker has no roofline model");
        }
        const double perf = sys->roofline->get_perf(1.0);
        if (!std::isfinite(perf) || perf <= 0.0) {
            delete completion_cookie;
            fail_closed("roofline control marker has degenerate performance");
        }
        double compute_elapsed = 1.0 / perf;
        if (sys->local_mem_latency > 0) {
            const double compute_only = 1.0 / sys->peak_perf;
            const double memory_elapsed =
                static_cast<double>(sys->local_mem_latency) / 1e9 +
                1.0 / sys->local_mem_bw;
            compute_elapsed = std::max(compute_only, memory_elapsed);
        }
        const double runtime_ns_f = compute_elapsed * 1e9;
        if (!std::isfinite(runtime_ns_f) || runtime_ns_f >= 1e18) {
            delete completion_cookie;
            fail_closed("roofline control marker runtime is not representable");
        }
        roofline_runtime = static_cast<uint64_t>(runtime_ns_f);
        if (node.compute.runtime_ns != 0) {
            roofline_runtime = node.compute.runtime_ns;
        }
    }

    if (local_hbm_bandwidth_model != nullptr &&
        (die_backend || node.compute.runtime_ns == 0)) {
        // This is the exact legacy marker branch for both Die and roofline
        // execution: a 1-op/1-byte COMPUTE fluid job, paid through the real
        // HBM latency/share model, with completion forwarded to the owning
        // transport sink by Workload::call.
        local_hbm_bandwidth_model->issue_compute(1, 1, completion_cookie);
        return;
    }

    const uint64_t runtime = die_backend ? uint64_t{1} : roofline_runtime;
    if (!node.is_cpu_op) hw_resource->tics_gpu_ops += runtime;
    sys->register_event(this, EventType::General, completion_cookie, runtime);
}

void Workload::issue_die_comp(const ExecutionDriven::NodeView& node) {
    // Contract §7.3 dispatch order (frozen 2026-10-03): is_timer_op -> old
    // timer path; is_cpu_op -> issue_replay; op_role=control_marker -> the
    // legacy micro-job semantics verbatim; die_spec -> DieModelAdapter
    // (runtime_ns must be 0); anything else -> fail-closed. The legacy
    // roofline closed form is NEVER reachable under the die backend.
    // Guards use this repository's fail_closed (abort) convention, not a
    // throw: the static path runs under Sys::call_events which swallows
    // std::exception and would strand the occupied node (FailClosed.hh).
    if (node.is_timer_op) {
        issue_replay(node);
        return;
    }
    if (node.is_cpu_op) {
        issue_replay(node);
        return;
    }
    if (node.compute.die_spec != nullptr && node.compute.runtime_ns != 0) {
        // The GraphBatch parser already rejects this combination (N12 /
        // contract §2); reaching it here means a producer that bypassed the
        // parser (e.g. a static ET view). Fail closed instead of letting a
        // calibrated runtime silently bypass the die execution.
        fail_closed(
            "die backend requires runtime_ns == 0 on nodes carrying a "
            "die_op_spec (node " +
            std::to_string(node.global_id) + ")");
    }
    if (node.compute.op_role == ExecutionDriven::OpRole::ControlMarker) {
        // Legacy micro-job semantics, byte-for-byte (contract §7.3): with
        // the HBM fluid model the marker drains a (1 op, 1 byte) COMPUTE
        // job (the model's own latency phase included); without it a 1 ns
        // event. No die macro-group launch, no plan, no PE/VEC stages.
        // (No MetricCollector on_compute_issue for markers: the 1/1 pair
        // is completion-path noise, not a second cost source.)
        WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
        wlhd->sys_id = sys->id;
        wlhd->workload = this;
        wlhd->node_id = node.global_id;
        if (local_hbm_bandwidth_model != nullptr) {
            local_hbm_bandwidth_model->issue_compute(1, 1, wlhd);
        } else {
            hw_resource->tics_gpu_ops += 1;
            sys->register_event(this, EventType::General, wlhd, 1);
        }
        return;
    }
    if (node.compute.die_spec == nullptr) {
        // Contract §2/§7.3: a real COMP without a die_op_spec is a
        // generator defect -- never a silent roofline/replay fallback.
        fail_closed(
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
    die_adapter_->issue_die(node);
}

void Workload::issue_comp(const ExecutionDriven::NodeView& node) {
    // Rollout stage 1 red line (design doc §6.4): the die backend must not
    // run the legacy closed-form roofline / fluid COMP path -- reaching here
    // means the COMP dispatch order regressed.
    if (sys->compute_cost_model == ComputeCostModel::Die) {
        fail_closed(
            "Workload::issue_comp (legacy roofline path) must never run "
            "under the die compute-cost-model backend");
    }
    // Contract guards: unreachable through the sole production caller
    // (Workload::issue routes !roofline_enabled / is_cpu_op to issue_replay)
    // and kept fail-closed rather than throwing -- Sys::call_events swallows
    // std::exception, which would strand the occupied node (FailClosed.hh).
    if (!this->sys->roofline_enabled) {
        fail_closed("Roofline model is not enabled for non-replay comp");
    }

    if (node.is_cpu_op) {
        fail_closed("Roofline is only available for GPU nodes");
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

    double operational_intensity = num_ops / tensor_size;
    double perf = sys->roofline->get_perf(operational_intensity);
    // Degenerate-roofline guard (fail-fast at the production site):
    // num_ops == 0 makes perf == min(bandwidth*0, peak_perf) == 0, and a
    // zero peak-perf/local-mem-bw (missing config keys) pins perf to 0 as
    // well. Division by that perf would yield 0/0 = NaN or x/0 = inf, which
    // then spreads three ways: the static_cast<uint64_t> below is UB, the
    // static path pollutes per-node statistics, and the online compact path
    // aborts at its isfinite gate. Fail closed (not throw): the node is
    // already taken/occupied/record-started at this point, and a throw from
    // this issue path would be swallowed by Sys::call_events and strand it
    // (FailClosed.hh).
    if (node_num_ops == 0 || !std::isfinite(perf) || perf <= 0.0) {
        fail_closed(
            "Roofline comp node has degenerate performance: num_ops=" +
            std::to_string(node_num_ops) +
            " tensor_size=" + std::to_string(node_tensor_size) +
            " perf=" + std::to_string(perf) +
            " (check num_ops in the workload and peak-perf/local-mem-bw in "
            "the system config)");
    }
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
            fail_closed(
                "HBM KV-restore sharing cannot be combined with remote "
                "operand pipeline loads on the same COMP node");
        }
        if (sys->remote_mem_bw <= 0) {
            fail_closed(
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

    // R5-Q5 fix (ported 2026-09-30): elapsed_time comes from user data
    // divided by configured rates; a pathological configuration can push
    // elapsed_time * 1e9 outside the uint64 range, where the floating ->
    // integer conversion is UB. Anything >= 1e18 ns (~31.7 simulated
    // years) cannot be a real run; fail closed instead.
    const double runtime_ns_f = elapsed_time * 1e9;
    if (!std::isfinite(runtime_ns_f) || runtime_ns_f >= 1e18) {
        fail_closed("computed comp runtime " + std::to_string(runtime_ns_f) +
                    " ns is not representable (elapsed_time " +
                    std::to_string(elapsed_time) +
                    " s); check peak-perf/local-mem-bw/remote-mem-bw and the "
                    "node's num_ops");
    }
    uint64_t runtime = static_cast<uint64_t>(runtime_ns_f);  // sec -> ns
    // Step 1-8: online execution-driven calibration. The online GraphBatch
    // (graph_batch_builder.py) carries planner-LUT-aligned durations so the
    // engine timeline is order-isomorphic to the Python decision sequence
    // (fail-closed order consumption). The static .et
    // path never sets runtime_ns (duration_micros=0, parsed as 0) and keeps
    // the roofline above -- byte-for-byte preserved.
    if (node.compute.runtime_ns != 0ul) {
        runtime = node.compute.runtime_ns;
    }
    if (local_hbm_bandwidth_model != nullptr &&
        node.compute.runtime_ns == 0ul) {
        // sh_3.0 HBM 50/50 execution model (preserved object #18): the model
        // owns the compute/restore job timing. A non-zero calibrated
        // runtime_ns never enters the model and keeps its calibrated
        // duration (calibration honor; 中-5 gate aligned 2026-08-20 with
        // the other four repos -- the five-repo priority contract is
        // calibration > fluid contention > roofline).
        local_hbm_bandwidth_model->issue_compute(node_num_ops,
                                                 node_tensor_size, wlhd);
    } else {
        // 中-4③: the HBM-model-closed fallback must keep the same is_cpu_op
        // discrimination as the other four repos (face repo issue_comp
        // semantics) instead of blanket-attributing to tics_gpu_ops (the
        // old sibling
        // tics_cpu_ops counter was write-only and was removed).
        if (!node.is_cpu_op) {
            hw_resource->tics_gpu_ops += runtime;
        }
        sys->register_event(this, EventType::General, wlhd, runtime);
    }

    if (execution_mode_ == ExecutionDriven::ExecutionMode::Online &&
        !stats->online_history_preserved()) {
        auto& online_stat = online_statistics_state_or_fail(node.global_id);
        online_stat.compute_utilization = perf / sys->peak_perf;
        online_stat.memory_utilization =
            (perf / operational_intensity) / sys->local_mem_bw;
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
        fail_closed("Comm node should not be on CPU");
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
        fail_closed("Unknown comm node type");
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
        // (The second set_notifier that used to sit here was identical to
        // the one above the registration -- a copy-paste leftover; the
        // setter already deletes-and-replaces, so removing it is a no-op.)
    } else {
        fail_closed("Unsupported collective comm type");
    }
}

void Workload::issue_send_comm(const ExecutionDriven::NodeView& node) {
    const auto src = node.comm.src;  // adapter default: this rank
    if (src != this->sys->id) {
        fail_closed("Send node should be issued by the sender");
    }
    const auto dst = node.comm.dst;
    const auto size = node.comm.bytes;
    const auto tag = node.comm.tag;

    // sim_request fields are write-only in this tree (the analytical
    // backend never reads them; face-LRU audit 2026-09-29 mirrored to
    // joint); the declaration stays for the front-end virtual interface.
    sim_request snd_req{};
    SendPacketEventHandlerData* sehd = new SendPacketEventHandlerData;
    sehd->callable = this;
    sehd->wlhd = new WorkloadLayerHandlerData;
    sehd->wlhd->node_id = node.global_id;
    sehd->event = EventType::PacketSent;
    // N-way HBM contention comm endpoint: the data sender reads the bytes
    // out of its own HBM (COMM_READ job starting at this node's issue
    // moment). The node completes on the join of the network-side
    // PacketSent and the HBM job. hbm-charge=false (NoC<->SerDes
    // pass-through at the edge rank) or zero bytes keeps the single-sided
    // legacy completion. Multi-hop transit ranks never issue a comm node
    // for the transfer, so they never touch their HBM (direct-through
    // routing assumption).
    if (hbm_endpoint_charge_active() && node.comm.hbm_charge && size > 0) {
        WorkloadLayerHandlerData* hbm_wlhd = begin_hbm_join(
            node.global_id, EventType::PacketSent);
        local_hbm_bandwidth_model->issue_comm_read(size, hbm_wlhd);
    }
    sys->front_end_sim_send(0, Sys::dummy_data, size, UINT8, dst, tag, &snd_req,
                            Sys::FrontEndSendRecvType::NATIVE,
                            &Sys::handleEvent, sehd);
}

void Workload::issue_recv_comm(const ExecutionDriven::NodeView& node) {
    const auto src = node.comm.src;
    const auto dst = node.comm.dst;  // adapter default: this rank
    if (dst != this->sys->id) {
        fail_closed("Recv node should be issued by the receiver");
    }
    const auto size = node.comm.bytes;
    const auto tag = node.comm.tag;

    sim_request rcv_req{};  // value-initialized (rendezvous copies the
                            // whole struct; partial init was formal UB)
    RecvPacketEventHandlerData* rcehd = new RecvPacketEventHandlerData;
    rcehd->wlhd = new WorkloadLayerHandlerData;
    rcehd->wlhd->node_id = node.global_id;
    rcehd->workload = this;
    rcehd->event = EventType::PacketReceived;
    // N-way HBM contention comm endpoint: the data receiver writes the
    // bytes into its own HBM (COMM_WRITE job at issue moment); completion
    // is the join of PacketReceived and the HBM job. The remote_load
    // NoC-stage recv at the KV target is annotated hbm-charge=false -- its
    // HBM write is carried by the following serial restore node, avoiding
    // double charging.
    if (hbm_endpoint_charge_active() && node.comm.hbm_charge && size > 0) {
        WorkloadLayerHandlerData* hbm_wlhd = begin_hbm_join(
            node.global_id, EventType::PacketReceived);
        local_hbm_bandwidth_model->issue_comm_write(size, hbm_wlhd);
    }
    sys->front_end_sim_recv(0, Sys::dummy_data, size, UINT8, src, tag, &rcv_req,
                            Sys::FrontEndSendRecvType::NATIVE,
                            &Sys::handleEvent, rcehd);
}

bool Workload::hbm_endpoint_charge_active() const {
    return sys->hbm_bandwidth_contention &&
        local_hbm_bandwidth_model != nullptr;
}

WorkloadLayerHandlerData* Workload::begin_hbm_join(
    uint64_t node_id,
    EventType terminal_event) {
    // One join state per node id; the local-HBM side is identified by its
    // wlhd pointer (the pointer is the callback cookie the
    // LocalHbmBandwidthModel hands back untouched), stored inside the live
    // join state (R4-13: no standalone pointer-keyed side map -- a side
    // that never fires leaves no dangling key behind). The network/port
    // side needs no registration: it is whatever cookie is not the stored
    // HBM-side one.
    WorkloadLayerHandlerData* hbm_wlhd = new WorkloadLayerHandlerData;
    hbm_wlhd->sys_id = sys->id;
    hbm_wlhd->workload = this;
    hbm_wlhd->node_id = node_id;
    const auto [it, inserted] = hbm_join_pending_.emplace(
        node_id,
        HbmJoinState{node_id, terminal_event, false, false, hbm_wlhd});
    if (!inserted) {
        delete hbm_wlhd;
        fail_closed("duplicate HBM endpoint join for node " +
                    std::to_string(node_id));
    }
    return hbm_wlhd;
}

bool Workload::consume_hbm_join_event(WorkloadLayerHandlerData* wlhd) {
    const uint64_t node_id = wlhd->node_id;
    auto state_it = hbm_join_pending_.find(node_id);
    if (state_it == hbm_join_pending_.end()) {
        // Fail-closed fire-after guard (R8-7): a re-delivery of the most
        // recently fired join's local-HBM-side cookie means the model fired
        // the same HBM job twice -- the ordinary terminal path would finish
        // an already-completed node a second time. Every other arrival
        // here is a normal single-sided completion (no join ever armed).
        if (reinterpret_cast<std::uintptr_t>(wlhd) ==
                last_fired_hbm_side_cookie_ &&
            node_id == last_fired_hbm_join_node_) {
            workload_logger_
                ->critical("HBM job completion for node {} has no pending "
                           "join entry (double fire or missing issue)",
                           node_id);
            exit(EXIT_FAILURE);
        }
        return false;  // not a joined side: normal single-sided completion
    }
    // Side classification against the live join's stored cookie (node ids
    // are globally unique and never reused, so keying by node id is
    // equivalent to the old per-cookie identity).
    const HbmJoinSide side =
        (wlhd == state_it->second.hbm_side_wlhd) ? HbmJoinSide::LocalHbm
                                                 : HbmJoinSide::NetworkPort;
    const std::uintptr_t hbm_side_cookie = reinterpret_cast<std::uintptr_t>(
        state_it->second.hbm_side_wlhd);
    delete wlhd;

    HbmJoinState& state = state_it->second;
    // Fail-closed same-side guard (R8-7): each side must arrive exactly
    // once; a same-side second arrival for a live join is a model-layer
    // mechanism violation (the flags below would otherwise be silently
    // re-set and the arrival swallowed).
    if (side == HbmJoinSide::NetworkPort) {
        if (state.network_done) {
            workload_logger_
                ->critical("duplicate network-side completion for node "
                           "id={}",
                           node_id);
            exit(EXIT_FAILURE);
        }
        state.network_done = true;
    } else {
        if (state.local_hbm_done) {
            workload_logger_
                ->critical("duplicate local-HBM job completion for node "
                           "id={}",
                           node_id);
            exit(EXIT_FAILURE);
        }
        state.local_hbm_done = true;
    }
    if (!state.network_done || !state.local_hbm_done) {
        return true;  // first side only; the node is still in flight
    }
    // Both sides fired: run the shared terminal body exactly once. The join
    // entry is erased first so the terminal body (which may re-enter
    // Workload::call synchronously) can never observe a stale join.
    const EventType terminal_event = state.terminal_event;
    last_fired_hbm_side_cookie_ = hbm_side_cookie;
    last_fired_hbm_join_node_ = node_id;
    hbm_join_pending_.erase(state_it);
    finish_general_node(node_id, terminal_event);
    return true;
}

void Workload::finish_general_node(uint64_t node_id, EventType event) {
    // Step 1-8: online mode has no ETFeederNode handle (et_node ==
    // nullptr); the online branch releases / records through the
    // NodeView. The static branch below stays byte-identical.
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
        const bool pipeline_anchor =
            nv->compute.exec_role ==
            ExecutionDriven::ComputeExecRole::RemoteReadPipelineAnchor;
        if (!pipeline_anchor) {
            hw_resource->release(*nv);
        }
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

    if (nv != nullptr) publish_ledger_transition("completion", *nv);

    // Fail closed, never throw: recordEnd runs under Sys::call_events
    // (static path), which swallows std::exception -- an escaping throw
    // (odd IO list, malformed size string, missing inputs/outputs attr)
    // would strand the bookkeeping and end the run silently incomplete.
    // Same convention as issue_comp.
    if (this->sys->track_local_mem &&
        (nv == nullptr || nv->compute.exec_role !=
                              ExecutionDriven::ComputeExecRole::RemoteReadPipelineAnchor)) {
        try {
            this->local_mem_usage_tracker->recordEnd(node,
                                                     Sys::boostedTick());
        } catch (const std::exception& e) {
            workload_logger_->critical(
                "local-mem tracker recordEnd failed (sys.id={}, node.id={}): "
                "{}",
                sys->id, node_id, e.what());
            exit(EXIT_FAILURE);
        }
    }

    // Step 1-3: unconditional node-terminal record (generic wlhd
    // branch; also reached by AnalyticalRemoteMemory and
    // LocalHbmBandwidthModel completions, which register_event back
    // into Workload::call -- sh_3.0 fourth-terminal audit: both
    // unique completion chains converge here); step 1-5
    // reverse-index fill in online mode. The N-way HBM join terminal
    // (comm / pool endpoints) converges here as well.
    record_node_terminal(execution_mode_, sys->id, node_id,
                         ExecutionDriven::NodeTerminalStatus::Success, nv);
    graph_source_->finish_node(node_id);
    if (nv != nullptr) publish_ledger_transition("release", *nv);
    if (execution_mode_ == ExecutionDriven::ExecutionMode::Online &&
        remote_read_executor_ != nullptr) {
        remote_read_executor_->notify_node_completed(sys->id, node_id);
    }
    // Static auto-advance only (see collective branch above).
    if (execution_mode_ == ExecutionDriven::ExecutionMode::Static) {
        issue_dep_free_nodes();
    }
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
    publish_ledger_transition("completion", node);
    // Step 1-4: the GraphSource is the sole dependency-state owner.
    graph_source_->finish_node(node_id);
    publish_ledger_transition("release", node);
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
    // Fail closed, never throw: recordEnd runs under Sys::call_events
    // (static path), which swallows std::exception -- an escaping throw
    // (odd IO list, malformed size string, missing inputs/outputs attr)
    // would strand the bookkeeping and end the run silently incomplete.
    // Same convention as issue_comp.
    if (this->sys->track_local_mem) {
        try {
            this->local_mem_usage_tracker->recordEnd(et_node,
                                                     Sys::boostedTick());
        } catch (const std::exception& e) {
            workload_logger_->critical(
                "local-mem tracker recordEnd failed (sys.id={}, node.id={}): "
                "{}",
                sys->id, node_id, e.what());
            exit(EXIT_FAILURE);
        }
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
            // R2 anchor fast path: online-only branch (the static else below is
            // untouched and stays unconditional); nv points into the NodeStore
            // record the registration hook set the flags on.
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

        // Fail closed, never throw: this recordEnd runs under
        // Sys::call_events (static path), which swallows std::exception;
        // an escaping throw would skip the terminal handling below and
        // strand the node (sim ends silently incomplete). Same convention
        // as issue_comp.
        if (this->sys->track_local_mem) {
            try {
                this->local_mem_usage_tracker->recordEnd(node,
                                                         Sys::boostedTick());
            } catch (const std::exception& e) {
                workload_logger_->critical(
                    "local-mem tracker recordEnd failed (sys.id={}, "
                    "node.id={}): {}",
                    sys->id, node_id, e.what());
                exit(EXIT_FAILURE);
            }
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
            if (wlhd->pipeline_sink != nullptr) {
                // A compact pipeline control marker borrows the legacy
                // Workload HBM COMPUTE job, but its logical endpoint is a
                // virtual hole rather than a NodeStore node. Transfer the
                // cookie exactly once to its owning transport/executor.
                Callable* const sink = wlhd->pipeline_sink;
                wlhd->pipeline_sink = nullptr;
                sink->call(event, data);
                return;
            }
            // N-way HBM contention join: a charged comm/pool node delivers
            // two wlhd-carrying completions (network/port side and HBM
            // side). Only after both fired does the node terminal path run
            // -- exactly once (idempotent by map erasure).
            if (!consume_hbm_join_event(wlhd)) {
                finish_general_node(wlhd->node_id, event);
                delete wlhd;
            }
        }
    }

    // Final end authority (方案 §4 步骤 1-2 操作 3): in online mode this
    // side effect is disabled -- a temporarily empty graph (including HBM
    // model idle windows) must never end the online simulation and drop
    // later injections. The end authority belongs exclusively to the
    // ServiceCoordinator (总体方案 §5.5). The static path keeps the
    // pre-phase-1 behavior byte-for-byte (including the sh-family
    // hbm_dma slot and has_active_jobs conditions).
    if (execution_mode_ != ExecutionDriven::ExecutionMode::Online) {
        // Step 1-4: static_all_done (free empty && ongoing empty) comes from
        // the GraphSource -- the sole dependency-state owner.
        if ((graph_source_->static_all_done()) &&
            (hw_resource->num_in_flight_cpu_ops == 0) &&
            (hw_resource->num_in_flight_gpu_comp_ops == 0) &&
            (hw_resource->num_in_flight_gpu_comm_ops == 0) &&
            (hw_resource->num_in_flight_hbm_dma_ops == 0) &&
            // 方案 §4: the static/ETFeeder sim-finish gate also counts the
            // remote-MEM class (counted, unlimited HardwareResource slot --
            // the former comm-single-slot fall-through is gone, so the gate
            // must follow the slot the nodes actually occupy).
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
             !local_hbm_bandwidth_model->has_active_jobs()) &&
            // Die backend (rollout stage 1), mirror of the note above: a
            // DieExecution completes its node through finish_general_node
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
}

void Workload::fire() {
    call(EventType::General, NULL);
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
    workload_logger_
        ->info("sys[{}] finished, {} cycles, exposed communication {} cycles.",
               sys->id, curr_tick, curr_tick - hw_resource->tics_gpu_ops);
    stats->post_processing();
    stats->report();
    // Fail closed, never throw: report() runs under Sys::call_events
    // (static path), which swallows std::exception -- a tracker throw here
    // (tmpfile/fwrite spool failure, fmt error) would skip
    // sim_notify_finished and hang the run-end, or end the simulation
    // silently without its local-mem outputs. Same convention as
    // issue_comp.
    if (this->sys->track_local_mem) {
        try {
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
        } catch (const std::exception& e) {
            workload_logger_->critical(
                "local-mem tracker report failed (sys.id={}): {}", sys->id,
                e.what());
            exit(EXIT_FAILURE);
        }
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
