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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdlib.h>

using namespace std;
using namespace AstraSim;
using namespace Chakra::FeederV3;
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

}  // namespace

Workload::Workload(Sys* sys, string et_filename, string comm_group_filename,
                   ExecutionDriven::ExecutionMode execution_mode,
                   std::shared_ptr<ExecutionDriven::GraphSource> graph_source) {
    this->workload_logger_ = LoggerFactory::get_logger("workload");
    this->graph_source_ = std::move(graph_source);
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
    if (sys->hbm_bandwidth_contention || sys->hbm_kv_restore_bandwidth_sharing) {
        // sh_2.0: the LocalHbmBandwidthModel is a preserved execution model
        // (contract B11) -- assembled in BOTH modes, identical timing/params.
        // With hbm-bandwidth-contention (default) it runs the N-way
        // equal-split policy over its HBM users (COMP, NoC p2p comm
        // endpoints, pool endpoints, and KV restore only under the
        // hbm-kv-restore-bandwidth-sharing switch -- 策略说明 §4.4, M24:
        // a sharing-off restore keeps the closed-form single-slot timing in
        // issue_local_hbm_kv_restore); with only the legacy
        // hbm-kv-restore-bandwidth-sharing flag (contention off) no comm/pool
        // jobs are ever issued, so the model degenerates to the historical
        // one-COMP-plus-one-restore 50/50 behavior (A/B baseline).
        this->local_hbm_bandwidth_model =
            std::make_unique<LocalHbmBandwidthModel>(sys, this);
    }
    this->sys = sys;
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
    // Online service statistics compact terminal records (the legacy
    // history-preserving mode is retained only for direct Statistics users,
    // e.g. the microbenchmark regression fixture).
    this->stats->configure_online_history_preservation(false);
}

Workload::~Workload() {
    // Stage B teardown FIRST: every live DieExecution cancels its pending
    // event and drops its pending HBM die job while all collaborators (the
    // fluid model, the graph source) are still alive (design doc §6.3: no
    // leak, no double completion, no dangling callbacks). The member order
    // already destroys die_adapter_ before local_hbm_bandwidth_model; this
    // explicit shutdown additionally covers the normal (non-throwing) path
    // before the fluid model's own drain below.
    if (this->die_adapter_ != nullptr) {
        this->die_adapter_->shutdown();
    }

    // 先销毁 HBM 流体模型，趁它能触碰的每个成员都还活着：模型析构器经
    // abandon_hbm_endpoint_join() 排空残余作业并从 hbm_endpoint_joins_
    // 抹除表项。hbm_endpoint_joins_ 在 Workload.hh 中的声明位置在
    // local_hbm_bandwidth_model 之后，隐式成员析构序会先拆 map 再跑模型
    // 自己的析构器——那次抹除会撞上已销毁的 map（异常拆卸路径上的
    // use-after-destruction UB：测试 harness 在通信作业仍在飞时删 Sys）。
    local_hbm_bandwidth_model.reset();

    // 方案 §3.4 早退纪律: this destructor OWNS the HBM endpoint join state
    // and must never silently swallow an unresolved join (a node whose port
    // and/or local-HBM leg never terminated). Report fail-closed -- the
    // cookies themselves belong to their issuing legs (the backend's
    // shutdown deletes only its own port cookies; the local-HBM model owns
    // its own) and are NOT cleaned here.（模型 drain 只抹除其端点作业
    // 仍在飞的表项；此处剩的是两腿都未终态的节点。）
    if (!hbm_endpoint_joins_.empty()) {
        workload_logger_->critical(
            "Workload sys.id={} destroyed with {} unresolved HBM endpoint "
            "join(s); the joined node(s) never reached whole-node terminal",
            sys != nullptr ? sys->id : -1, hbm_endpoint_joins_.size());
    }
    comm_groups.clear();

    // Abort-path ownership（2026-10-01 对齐）：仍在飞的集合通信 DataSet
    // 归本 map 所有（正常完成路径 Workload::call 逐条删除并 erase）。
    // 没有这个循环，任何在集合通信中被拆除的 Workload（上游构造失败、
    // 测试 harness 下的 fatal 退出路径）都会泄漏全部在飞 DataSet。
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

void Workload::abandon_hbm_endpoint_join(uint64_t node_id) {
    // Teardown-only（见头文件声明）：抹除挂起的 join；不投递完成、不触发
    // 指标锚点——进程已在 fatal/teardown 路径上。
    hbm_endpoint_joins_.erase(node_id);
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
    if (!graph_source_->mark_terminal_observed(node_id)) {
        workload_logger_->critical(
            "duplicate, unknown, or unissued online terminal callback for "
            "node {}",
            node_id);
        std::exit(EXIT_FAILURE);
    }
}

// record_network_bandwidth was removed: the whole comm-size ->
// achieved-bandwidth chain had no production consumer (the computed
// per-node durations were dropped on the floor at every call site).  The
// compact-mode terminal fail-closed guard lives on in
// mark_online_terminal_or_fail / complete_online_statistics, which already
// ran on every terminal path before this function was ever reached.

void Workload::initialize_comm_groups(string comm_group_filename) {
    // communicator group input file is not given. Exact-match the sentinel:
    // the former "empty" substring test silently skipped any real path that
    // merely contains it (e.g. .../empty_dir/comm.json).
    if (comm_group_filename == "empty") {
        comm_groups.clear();
        return;
    }

    // Fail closed with a diagnostic instead of an escaping exception: this
    // runs in the Workload constructor ("new Workload" inside Sys, no
    // try/catch up the chain), so a missing/unreadable file, a malformed
    // document, or a non-numeric group key would otherwise surface as a
    // bare std::terminate at startup (wl-metrics-E3, aligned with face-LRU).
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

        // The CommunicatorGroup constructor validates its inputs and throws
        // std::invalid_argument (empty / duplicate ranks, non-positive id,
        // dimension sizes inconsistent with the rank count). This runs inside
        // the Workload constructor with nothing above catching, so a bare
        // throw would surface as std::terminate with no diagnosis. Fail
        // closed through the same channel as the file/parse guards above.
        try {
            comm_groups[comm_group_id] = std::make_shared<CommunicatorGroup>(
                comm_group_id, involved_NPUs, sys, dimension_sizes);
        } catch (const std::exception& e) {
            workload_logger_->critical(
                "communicator group file {}: failed to build group {} "
                "({} rank(s)): {}",
                comm_group_filename, comm_group_id, involved_NPUs.size(),
                e.what());
            exit(EXIT_FAILURE);
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
    if (pg_info.size() < 4) {
        // Fail closed on malformed metadata (wl-metrics-E1, aligned with
        // face-LRU): the strip-two-chars-from-each-side substr below needs
        // at least four bytes, and with size 1 the size()-4 underflow makes
        // substr's pos exceed the string and throw std::out_of_range from
        // OUTSIDE the try block below -- the issuing node is already
        // take_node'd/occupied, so the escaped exception would strand it.
        workload_logger_->critical(
            "malformed pytorch pg_info metadata (size {} < 4, rank {})",
            pg_info.size(), sys->id);
        exit(EXIT_FAILURE);
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

            // Trailing-garbage pg names must not parse as a silent prefix
            // ("12abc" is not group 12): the malformed name throws into the
            // non-fatal catch below, skipping the whole metadata payload --
            // the documented contract for this latent path.
            size_t pg_name_parsed_chars = 0;
            const int32_t pgNameInt =
                static_cast<int32_t>(std::stoi(pgName, &pg_name_parsed_chars));
            if (pg_name_parsed_chars != pgName.size()) {
                throw std::invalid_argument("pg_name " + pgName +
                                            " has trailing characters");
            }
            // Dedup must honor BOTH stored-id conventions, and it must not
            // depend on a same-definition comparison with a default empty
            // dimension list (the former CommunicatorGroup::matches_definition
            // helper, removed 2026-10-01): that comparison is always false
            // against a file-declared shaped group, which let an unshaped
            // torch re-declaration silently displace the shaped group -- the
            // exact opposite of the protection this guard claims. The torch
            // payload carries no dimensions, so whenever the ranks match the
            // incoming group can only be an equal or strictly less specific
            // re-declaration: skip both cases and keep the stored group.
            // File-loaded groups keep map key == get_id(), while torch-issued
            // groups are stored under pg_name with get_id() == pg_name + 1
            // ("to ensure pgName > 0"); both conventions count as "already
            // stored".
            auto existing = this->comm_groups.find(pgNameInt);
            if (existing != this->comm_groups.end() && existing->second &&
                existing->second->involved_NPUs == involved_NPUs &&
                (existing->second->get_id() == pgNameInt + 1 ||
                 existing->second->get_id() == pgNameInt)) {
                continue;
            }
            // To ensure pgName > 0
            auto cg = std::make_shared<CommunicatorGroup>(
                pgNameInt + 1, involved_NPUs, sys);
            this->comm_groups[pgNameInt] = cg;
        }
    } catch (const std::exception& e) {
        // Production graphs carry no METADATA node, so this catch guards a
        // latent path; the failure stays non-fatal by contract and is
        // reported through the workload logger instead of std::cerr.
        workload_logger_->warn(
            "rank {} failed to parse or process pytorch pg metadata: {}",
            sys->id, e.what());
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
        // Step 1-8: availability check on the NodeView. Strategy mode keeps
        // the real serialized physics (path-2 removal 2026-08-18: the
        // replay-only concurrent calibrated-COMP bypass was deleted with
        // the replay route).
        if (hw_resource->is_available(nv)) {
            issue(nv);
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
    this->hw_resource->occupy(node);
    // stats->record_end will be called in Workload::call
    start_online_statistics(node, Sys::boostedTick());
    // Side-band metrics observation only; does not touch the node, the
    // dependency resolver, or the event queue (doc sec.5.5).
    // R2 (2026-08-29) anchor fast path: consults the sparse NodeView flag
    // set by the B-1.5 registration hook; unanchored nodes (the
    // overwhelming majority) skip the two-level hash lookup entirely.
    if (MetricCollector::instance().enabled() && node.metric_issue_anchor) {
        MetricCollector::instance().on_node_issue(sys->id, node.global_id,
                                                  Sys::boostedTick());
    }
    // Path-2 removal follow-up（2026-10-01 对齐母仓 2026-09-26 的清除）：
    // sys->replay_only 的整节点 replay 分派已删除——那是随
    // --online-mode replay 一起退役的路由（issue_replay 算子路径保留，
    // 见下方 COMP 无 roofline / CPU op 分支）；下方节点类型 switch 是
    // 唯一分派。手写 system.json 的 "replay-only" 键仍被白名单接受并
    // 解析（模板兼容），但不再驱动任何分派。
    if ((node.node_type == ChakraNodeType::MEM_LOAD_NODE) ||
        (node.node_type == ChakraNodeType::MEM_STORE_NODE)) {
        if (node.is_local_hbm_kv_restore) {
            issue_local_hbm_kv_restore(node);
        } else {
            issue_remote_mem(node);
        }
    } else if (node.node_type == ChakraNodeType::COMP_NODE) {
        if (sys->compute_cost_model == ComputeCostModel::Die) {
            // Die backend (stage B, contract §7.3): the frozen dispatch
            // order lives in issue_die_comp; the legacy roofline/fluid path
            // below is unreachable under this backend.
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

void Workload::issue_metadata(const ExecutionDriven::NodeView& node) {
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
    sys->register_event(this, EventType::General, wlhd, runtime);
}

void Workload::issue_die_comp(const ExecutionDriven::NodeView& node) {
    // Contract §7.3 dispatch order (frozen 2026-10-03): is_timer_op -> old
    // timer path; is_cpu_op -> issue_replay; op_role=control_marker -> the
    // legacy micro-job semantics verbatim; die_spec -> DieModelAdapter
    // (runtime_ns must be 0); anything else -> fail-closed. The legacy
    // roofline closed form is NEVER reachable under the die backend.
    // (The MEM_LOAD/MEM_STORE dispatch above -- issue_local_hbm_kv_restore
    // / issue_remote_mem with their pipeline/join semantics -- is upstream
    // of this function and untouched by the die backend.)
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
        // parser. Fail closed instead of letting a calibrated runtime
        // silently bypass the die execution.
        throw std::runtime_error(
            "die backend requires runtime_ns == 0 on nodes carrying a "
            "die_op_spec (node " +
            std::to_string(node.global_id) + ")");
    }
    if (node.compute.op_role == ExecutionDriven::OpRole::ControlMarker) {
        // Legacy micro-job semantics, byte-for-byte (contract §7.3): with
        // the HBM fluid model the marker drains a (1 op, 1 byte) COMP job
        // (the model's own latency phase included); without it a 1 ns
        // event. No die macro-group launch, no plan, no PE/VEC stages.
        WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
        wlhd->sys_id = sys->id;
        wlhd->workload = this;
        wlhd->node_id = node.global_id;
        if (local_hbm_bandwidth_model != nullptr) {
            local_hbm_bandwidth_model->issue_compute(1, 1, wlhd);
        } else {
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
    die_adapter_->issue_die(node);
}

void Workload::issue_remote_mem(const ExecutionDriven::NodeView& node) {
    // sh_2.0 N-way HBM contention: a MEM node marked hbm-access-mode
    // (1 = local HBM pool read; bytes = tensor_size) is a pool traffic
    // endpoint -- its local HBM job competes in the N-way model and the node
    // completes as the join of the remote-port transaction and the HBM job
    // (hbm_endpoint_joins_ latch, see Workload::call).
    // Emitter premise (verified 2026-10-01 against the Python side, doc
    // sec.4.4): only the DIRECT chain-B mem_store (source rank == edge rank,
    // generate_wsc_llm_trace.py's single-mem_store path) emits
    // hbm_access_mode=1; the chain-A edge mem_store (remote store relayed
    // through the edge) always omits the key (=0, local zero-billing, port
    // fluid timing only) -- locked by test_kv_transfer_emission.py. The
    // protocol domain is {0, 1}: the commit boundary (GraphBatchCommitter
    // validate/preflight) fail-closes every other value. Tightened
    // (2026-10-01) from the old {0,1,2} wording: no mode=2 emitter exists,
    // so the pool-WRITE leg below has no protocol-reachable producer -- it
    // survives only for programmatic callers below the commit boundary
    // (the local_hbm_model_test harness drives it to exercise the write
    // leg of the N-way model). If a mode=2 emitter ever appears it must
    // come with its own doc-section ruling, otherwise the local pool WRITE
    // billing below would contradict the "chain-A edge is locally
    // zero-billed" contract.
    if (sys->hbm_bandwidth_contention &&
        local_hbm_bandwidth_model != nullptr &&
        node.hbm_access_mode > 0 && node.compute.tensor_size > 0) {
        hbm_endpoint_joins_[node.global_id] = HbmEndpointJoinState{};
        // 方案 §3.4 (SerDes 并发化后端, 2026-09-24): the two join legs carry
        // SEPARATE cookies. The local-HBM leg cookie belongs to the
        // Workload/LocalHbm owner (LocalHbmBandwidthModel keeps it until the
        // job retires and delivers it asynchronously through the model's
        // transition-event callback -- retire_completed_jobs ->
        // workload->call(General, wlhd) at a later tick -- and never deletes
        // it; the arriving Workload::call does); the port-leg cookie is the
        // backend wlhd solely owned by AnalyticalRemoteMemory's PortJob
        // (unique_ptr) until it is handed back through a Sys event. Sharing
        // ONE wlhd across both legs would let an early backend shutdown()
        // delete a pointer the HBM model still holds. Workload::call deletes
        // each arriving cookie on its own arrival; the latch below stays the
        // single whole-node terminal.
        WorkloadLayerHandlerData* hbm_wlhd = new WorkloadLayerHandlerData;
        hbm_wlhd->sys_id = sys->id;
        hbm_wlhd->workload = this;
        hbm_wlhd->node_id = node.global_id;
        if (node.hbm_access_mode == 1) {
            local_hbm_bandwidth_model->issue_pool_read(
                node.compute.tensor_size, hbm_wlhd);
        } else {
            local_hbm_bandwidth_model->issue_pool_write(
                node.compute.tensor_size, hbm_wlhd);
        }
    }
    WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
    wlhd->sys_id = sys->id;
    wlhd->workload = this;
    wlhd->node_id = node.global_id;
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
    // Path-2 removal (2026-08-18): the replay-only instant (1ns) HBM restore
    // completion branch was deleted with the replay route; strategy/static
    // keep the real LocalHbmBandwidthModel DMA physics.
    // M24 (策略说明 §4.4): restore joins the shared HBM model only under
    // the hbm-kv-restore-bandwidth-sharing switch -- with contention on but
    // sharing off it keeps the closed-form single-slot timing below instead
    // of silently entering the N-way model (the switch used to be masked by
    // the `contention || sharing` model-construction gate).  A zero-byte
    // restore never creates an HBM job (same Workload-layer guard as the
    // other three HBM issue paths) and falls through to the fixed-latency
    // closed form instead of throwing from inside the model, where Sys::
    // call_events would fail closed on the escaping exception.
    if (local_hbm_bandwidth_model != nullptr &&
        sys->hbm_kv_restore_bandwidth_sharing && tensor_size > 0) {
        local_hbm_bandwidth_model->issue_restore(tensor_size, wlhd);
        return;
    }

    // Closed-form single-slot timing (both HBM switches off, sharing off, or
    // a zero-byte restore): fail closed on a missing/non-positive
    // local-mem-bw instead of dividing into an Inf runtime cast to uint64
    // (M14; the Sys zero-rate guard only clears the contention flag and
    // never protects this fallback division).
    if (sys->local_mem_bw <= 0) {
        workload_logger_->critical(
            "local HBM KV restore requires local-mem-bw in the system "
            "configuration (local_mem_bw={})",
            sys->local_mem_bw);
        exit(EXIT_FAILURE);
    }
    const double elapsed_seconds =
        (static_cast<double>(sys->local_mem_latency) / 1e9) +
        static_cast<double>(tensor_size) / sys->local_mem_bw;
    // R5-Q5 fix (2026-10-01, ported from face-LRU): elapsed_seconds comes
    // from configured sizes divided by configured rates; a pathological
    // configuration (tiny but positive local-mem-bw) can push
    // elapsed_seconds * 1e9 outside the uint64 range, where the floating ->
    // integer conversion is UB. Anything >= 1e18 ns (~31.7 simulated years)
    // cannot be a real run; fail closed instead.
    const double runtime_ns = elapsed_seconds * 1e9;
    if (!std::isfinite(runtime_ns) || runtime_ns >= 1e18) {
        workload_logger_->critical(
            "computed HBM KV-restore fallback runtime {} ns is not "
            "representable (elapsed_seconds {} s); check local-mem-bw and "
            "the node's tensor size",
            runtime_ns, elapsed_seconds);
        exit(EXIT_FAILURE);
    }
    const uint64_t runtime = std::max<uint64_t>(
        1, static_cast<uint64_t>(std::ceil(runtime_ns)));
    sys->register_event(this, EventType::General, wlhd, runtime);
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
    // issue() dispatch chain (wl-metrics-E2, aligned with face-LRU): the
    // node was already taken/occupied by the time we run, and an invariant
    // breach surfacing as a throw would kill the run from inside the
    // Sys::call_events dispatcher (which fails closed on any escaping
    // exception). Any invariant breach must therefore be fail-closed
    // (critical + exit) at the source with a scoped diagnosis, not throw.
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

    // Fail-closed configuration check (aligned with the face fix of its M9):
    // roofline is enabled, but its two divisors below were never configured
    // (Sys defaults both to 0 when the "peak-perf"/"local-mem-bw" keys are
    // missing or non-positive).  Using them would produce NaN/Inf perf or
    // utilization values.
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
            // The remote-operand pipeline roofline and the shared-HBM fluid
            // model are two different timing authorities for the same COMP
            // node; combining them is a configuration error, not a data
            // property.  (This repository materializes traces with
            // remote_operand_loads=false, so the branch is unreachable in
            // the standard pipeline -- kept fail-closed for parity; critical
            // + exit per the wl-metrics-E2 note at the top of this
            // function.)
            workload_logger_->critical(
                "HBM bandwidth contention cannot be combined with remote "
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

    // sec -> ns, rounded up with a 1ns floor (wl-metrics-C2, aligned with
    // face-LRU) -- the same conversion convention as the restore closed-form
    // fallback in issue_local_hbm_kv_restore and the HBM model's fluid
    // transition scheduler (never a 0ns event).
    // R5-Q5 fix (2026-10-01, ported from face-LRU): elapsed_time comes from
    // user data divided by configured rates; a pathological configuration
    // (tiny but positive peak_perf) can push elapsed_time * 1e9 outside the
    // uint64 range, where the floating -> integer conversion is UB. Anything
    // >= 1e18 ns (~31.7 simulated years) cannot be a real run; fail closed
    // instead.
    const double runtime_ns = elapsed_time * 1e9;
    if (!std::isfinite(runtime_ns) || runtime_ns >= 1e18) {
        workload_logger_->critical(
            "computed comp runtime {} ns is not representable (elapsed_time "
            "{} s); check peak-perf/local-mem-bw and the node's num_ops",
            runtime_ns, elapsed_time);
        exit(EXIT_FAILURE);
    }
    uint64_t runtime = std::max<uint64_t>(
        1, static_cast<uint64_t>(std::ceil(runtime_ns)));
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
        // Falls through to the shared operator-statistics tail below.
        wlhd->sys_id = sys->id;
        wlhd->workload = this;
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
                        online_stat.memory_utilization.value_or(0),
                        tensor_size, num_ops);
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
                        op_stat.memory_utilization.value_or(0), tensor_size,
                        num_ops);
        }
    }
}

void Workload::issue_comm(const ExecutionDriven::NodeView& node) {
    // Fail-closed, not throw (wl-metrics-E2): see issue_comp -- an escaping
    // throw would kill the run inside the Sys::call_events dispatcher
    // instead of reporting a scoped diagnosis.
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
    // involved_dim is an optional coll attribute in ParsedGraphBatch
    // (absent == empty vector). The online commit preflight requires it to
    // be non-empty and match the declared dimension count; Sys::
    // generate_collective additionally pads a short vector to the topology
    // dimension count (N14) for callers that bypass that preflight.
    const std::vector<bool>& involved_dims = node.coll.involved_dim;

    auto comm_group_owner = extract_comm_group(node);
    CommunicatorGroup* comm_group = comm_group_owner.get();
    const auto comm_type =
        static_cast<ChakraCollectiveCommType>(node.coll.comm_type);
    const auto comm_size = node.coll.bytes;
    const auto comm_priority = node.coll.priority;  // default 0u

    // G2-6: the four implemented collectives share an identical epilogue
    // (retain the group owner / record the reverse maps / arm the
    // notifier); only the generate call differs, so each branch below just
    // assigns fp and the epilogue runs once at the end.
    DataSet* fp = nullptr;
    if (comm_type == ChakraCollectiveCommType::ALL_REDUCE) {
        fp = sys->generate_all_reduce(comm_size, involved_dims,
                                      comm_group, comm_priority, node.global_id);
    } else if (comm_type == ChakraCollectiveCommType::ALL_TO_ALL) {
        fp = sys->generate_all_to_all(comm_size, involved_dims,
                                      comm_group, comm_priority, node.global_id);
    } else if (comm_type == ChakraCollectiveCommType::ALL_GATHER) {
        fp = sys->generate_all_gather(comm_size, involved_dims,
                                      comm_group, comm_priority, node.global_id);
    } else if (comm_type == ChakraCollectiveCommType::REDUCE_SCATTER) {
        fp = sys->generate_reduce_scatter(comm_size, involved_dims,
                                          comm_group, comm_priority, node.global_id);
    } else if (comm_type == ChakraCollectiveCommType::BROADCAST) {
        // Broadcast has no collective implementation here.  The online commit
        // boundary rejects BROADCAST graphs; reaching this branch means an
        // unvalidated input slipped past it (static ET path or a bypassed
        // preflight).  Fail closed (critical + exit, not throw: callers run
        // under Sys::call_events, which fails closed on any escaping
        // exception -- wl-metrics-E2 -- so the source-level exit keeps the
        // diagnosis scoped to this branch instead of a generic dispatcher
        // abort)
        // instead of replaying the node as a local event with no
        // communication, network cost, or collective semantics.
        workload_logger_->critical(
            "BROADCAST collective is not implemented; broadcast graphs must "
            "be rejected at the commit boundary (sys.id={}, node={})",
            sys->id, node.global_id);
        exit(EXIT_FAILURE);
    } else {
        // Same fail-closed discipline as the BROADCAST branch above.
        workload_logger_->critical(
            "Unsupported collective comm type (sys.id={}, node={}, "
            "comm_type={})",
            sys->id, node.global_id, static_cast<int>(comm_type));
        exit(EXIT_FAILURE);
    }
    fp->retain_communicator_group(comm_group_owner);
    collective_comm_node_id_map[fp->my_id] = node.global_id;
    collective_comm_wrapper_map[fp->my_id] = fp;
    fp->set_notifier(this, EventType::CollectiveCommunicationFinished);
}

void Workload::issue_send_comm(const ExecutionDriven::NodeView& node) {
    const auto src = node.comm.src;  // adapter default: this rank
    if (src != this->sys->id) {
        // Fail-closed, not throw (wl-metrics-E2): see issue_comp.
        workload_logger_->critical(
            "Send node should be issued by the sender");
        exit(EXIT_FAILURE);
    }
    const auto dst = node.comm.dst;
    const auto size = node.comm.bytes;
    const auto tag = node.comm.tag;

    // sh_2.0 N-way HBM contention: the p2p sender is a data endpoint of its
    // own HBM (reads comm bytes out of local HBM). hbm-charge=false marks
    // pass-through traffic (NoC<->SerDes relay at an edge rank); multi-hop
    // transit never issues a node on the passed-through rank at all. The
    // node completes as the join of the network packet event and the local
    // HBM job (hbm_endpoint_joins_ latch, see Workload::call).
    WorkloadLayerHandlerData* wlhd = nullptr;
    if (sys->hbm_bandwidth_contention &&
        local_hbm_bandwidth_model != nullptr && size > 0 &&
        node.comm.hbm_charge) {
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
        wlhd->sys_id = sys->id;
        wlhd->workload = this;
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
        // Fail-closed, not throw (wl-metrics-E2): see issue_comp.
        workload_logger_->critical(
            "Recv node should be issued by the receiver");
        exit(EXIT_FAILURE);
    }
    const auto size = node.comm.bytes;
    const auto tag = node.comm.tag;

    // sh_2.0 N-way HBM contention: the p2p receiver is a data endpoint of
    // its own HBM (writes comm bytes into local HBM). hbm-charge=false marks
    // pass-through (NoC->SerDes relay at an edge rank) or the remote-load
    // target whose HBM write is already carried by the serially-following
    // restore node (single charge per byte flow).
    WorkloadLayerHandlerData* wlhd = nullptr;
    if (sys->hbm_bandwidth_contention &&
        local_hbm_bandwidth_model != nullptr && size > 0 &&
        node.comm.hbm_charge) {
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
        wlhd->sys_id = sys->id;
        wlhd->workload = this;
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

        const ExecutionDriven::NodeView* nv = graph_source_->lookup_ptr(node_id);
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

        graph_source_->finish_node(node_id);

        // Sole-owner reclamation of the finished DataSet. Lifecycle evidence
        // that nothing can still hold a dangling pointer at this erase point:
        // (1) the per-stream raw `dataset` member is read exactly once, by
        // Sys::proceed_to_next_vnet_baseline at final-phase teardown, and the
        // deleting notification fires only after EVERY stream has passed that
        // point (finished_streams == total_streams), so no live stream can
        // touch it afterwards; (2) the degenerate zero-stream path registers
        // the DataSet itself as its completion event, and Sys::call_events
        // pops the node before invoking it, so re-entrancy cannot free it
        // mid-call, and DataSet::notify_stream_finished touches no member
        // after the notifier returns (only the local IntData); (3) the
        // notifier fires exactly once (nulled before invocation) and this
        // map's entries were erased above, so a duplicate delivery fails
        // closed on lookup instead of reaching this delete twice; (4) the
        // event loop is single-threaded, so no completion can interleave with
        // issue_coll_comm's insert + set_notifier. A fully completed run
        // leaves this map empty (the phase-4 terminal audit gates on every
        // committed node reaching terminal before the main loop breaks);
        // abnormal paths exit() without teardown, so no residual entry is
        // ever destroyed while a pending event could still reference it.
        // ~DataSet releases communicator_group_owner_, ending the collective's
        // pin on the CommunicatorGroup object graph.
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
            // sh_2.0 N-way HBM contention endpoint join: a comm send/recv or
            // pool MEM node carrying a local HBM job completes only when BOTH
            // async sides have called back (network packet / remote-port
            // backend event AND the LocalHbmBandwidthModel job). The first
            // arrival only consumes its own cookie, decrements the latch and
            // returns: the node stays in flight (no release / record_end /
            // finish_node), and the second arrival runs the terminal handling
            // below exactly once (idempotent per arrival, either order).
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
                    // 方案 §3.4 cookie 所有权: a remote-MEM join (its
                    // completion_event is left at the General default) gives
                    // each leg its own cookie -- the arriving one is owned by
                    // THIS arrival and is deleted here; the partner leg's
                    // cookie is deleted by its own arrival. The legacy p2p
                    // comm joins (completion_event PacketSent/PacketReceived)
                    // still share ONE wlhd between the network frontend
                    // handler and the local-HBM job, and neither ever deletes
                    // it mid-flight, so their first arrival must NOT delete
                    // -- the latch-opening arrival deletes the shared cookie
                    // at the terminal below.
                    if (hbm_join_event == EventType::General) {
                        delete wlhd;
                    }
                    return;
                }
                hbm_endpoint_joins_.erase(join_it);
            }
            finish_generic_node(wlhd->node_id);
            // Exactly-once cookie consumption at the latch-opening arrival:
            // a remote-MEM join deletes THIS arrival's own leg cookie, a
            // comm join deletes the single shared wlhd.
            delete wlhd;
        }
    }
}

void Workload::finish_generic_node(uint64_t node_id) {
    // Extracted verbatim from the former generic wlhd branch of
    // Workload::call: node release / stats / metrics / terminal record /
    // dependency release.  Shared by the plain register_event completions,
    // the LocalHbmBandwidthModel COMP completions, and the joined p2p comm
    // completions.
    const ExecutionDriven::NodeView* nv = graph_source_->lookup_ptr(node_id);
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
    // R2 anchor fast path: nv points into the NodeStore record the
    // registration hook set the flags on.
    if (MetricCollector::instance().enabled() &&
        nv->metric_complete_anchor) {
        MetricCollector::instance().on_node_complete(
            sys->id, node_id, Sys::boostedTick());
    }
    // R3 (方案 §3.6 / 阶段 E, 2026-08-29), as carried by the SerDes
    // concurrent-port backend (方案 §3.1-3.4, 2026-09-24): per-
    // transaction remote-memory accounting lives entirely in the
    // backend -- the [H4] stats hook
    // AnalyticalRemoteMemory::on_stats_batch_delivery settles each
    // delivered batch before any Workload callback, and the
    // per-transaction detail rows (keyed by the issue-time
    // detail_sys_id/detail_node_id copies) land in
    // remote_memory_transactions.jsonl. Accounting here would repeat
    // the two old defects: the key was sys_id (a shared port's queue
    // split into per-rank virtual ledgers) and the record moment was
    // the node terminal, which an HBM join can hold past the real
    // port-transaction completion.

    // (The former p2p network-bandwidth accounting block was removed with
    // record_network_bandwidth: the per-node durations it computed were
    // dropped on the floor by the no-op callee.)

    // Step 1-3: unconditional node-terminal record (generic wlhd branch;
    // also reached by AnalyticalRemoteMemory completions, which
    // register_event back into Workload::call); step 1-5 reverse-index fill.
    record_node_terminal(sys->id, node_id,
                         ExecutionDriven::NodeTerminalStatus::Success, nv);

    graph_source_->finish_node(node_id);
}

void Workload::record_die_utilizations(const uint64_t node_id,
                                       const double compute_utilization,
                                       const double memory_utilization,
                                       const double vector_utilization) {
    // Contract §7.6: the die-only vector_utilization lives on the compact
    // OnlineStatisticsState; the legacy OperatorStatistics record
    // (history-preserving microbenchmarks) has no vector field, so only
    // compute/memory cross over there.
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

    // Non-numeric pg_name must not throw (wl-metrics-E2, aligned with
    // face-LRU): a throw here would leave the already-issued node
    // occupied forever. Fail closed with a full-string validation instead.
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
