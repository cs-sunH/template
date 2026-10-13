/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/**
die_execution_lifecycle_test.cc -- DieExecution end-to-end lifecycle
regression (stage B; hermetic online fixture, no sh_test_mesh artifacts).

Same fixture shape as die_model_adapter_test.cc (online Sys over a
NodeStoreGraphSource, fake backend on a real EventQueue, default die point
of contract §3.5). The Timer/Memory mixed plan is a real compiled `norm`
plan driven to Done.

The contention-off oracle is an in-test full-speed driver of the same plan
(same shape as the stage A3 execution parity driver: Timer -> advance(ns),
Memory -> advance(max(1, ceil(service_units * 1e9 / B))) + complete), so the
expected machine clock is computed independently of the runtime under test.

Cases:
  A. contention off + local_mem_latency 100: the wall span equals the
     oracle machine elapsed + exactly one 100 ns latency charge (the
     one-shot first-access TIMER); the terminal fires exactly once; the GPU
     comp slot is occupied during and released after the run (this repo
     removed the write-only tics_gpu_ops accumulator); the adapter's drain
     set empties.
  B. contention off + local_mem_latency 0: wall span == oracle machine
     elapsed exactly (no latency charge when the configured latency is 0).
  C. contention on: mid-flight the adapter reports an active execution and
     the GPU slot stays occupied; after the drain both clear; the span is
     at least estimate_alone + latency; the die side-band counters recorded
     service units AND actual bytes as separate quantities.
  D. marker under contention off: the legacy 1 ns event micro-job (issue
     tick + 1).
  E. teardown: an in-flight execution (big plan, partially pumped) is torn
     down with its Sys without crash, without any terminal firing, and the
     stale outer alarms drain harmlessly afterwards.

Build: cmake target
      AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest.
Exit code 0 on ALL PASS.
*/

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/HardwareResource.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/CoarseDieRuntime.hh"
#include "astra-sim/workload/DieModelAdapter.hh"
#include "astra-sim/workload/MetricCollector.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "DieModel.hh"
#include <astra-network-analytical/common/EventQueue.h>

#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace AstraSim;
using namespace AstraSim::ExecutionDriven;
using namespace NetworkAnalytical;

namespace {

bool g_ok = true;

void expect_true(const bool cond, const char* const what) {
    std::printf("%-72s -> %s\n", what, cond ? "PASS" : "FAIL");
    if (!cond) {
        g_ok = false;
    }
}

std::map<uint64_t, uint64_t> g_terminal_fires;
std::map<uint64_t, uint64_t> g_first_tick;

void tick_hook(void* /*ctx*/, int /*rank*/, uint64_t node_id,
               const char* /*request_id*/, const char* /*stage*/,
               uint64_t /*generation*/, uint64_t tick, int status) {
    if (status != static_cast<int>(NodeTerminalStatus::Success)) {
        std::fprintf(stderr,
                     "[die_execution_lifecycle_test] node %llu terminal "
                     "status %d\n",
                     static_cast<unsigned long long>(node_id), status);
        g_ok = false;
    }
    if (g_first_tick.find(node_id) == g_first_tick.end()) {
        g_first_tick[node_id] = tick;
    }
    ++g_terminal_fires[node_id];
}

class FakeQueueNetworkApi final : public AstraNetworkAPI {
  public:
    explicit FakeQueueNetworkApi(std::shared_ptr<EventQueue> queue)
        : AstraNetworkAPI(0), queue_(std::move(queue)) {}

    void complete_network_callbacks(bool enabled) {
        complete_network_callbacks_ = enabled;
    }

    int sim_send(void*, uint64_t, int, int, int, sim_request*,
                 void (*callback)(void*), void* data) override {
        if (complete_network_callbacks_)
            queue_->schedule_event(queue_->get_current_time() + 1, callback, data);
        return 0;
    }

    int sim_recv(void*, uint64_t, int, int, int, sim_request*,
                 void (*)(void*), void*) override {
        return 0;
    }

    void sim_schedule(const timespec_t delta, void (*fun_ptr)(void*),
                      void* fun_arg) override {
        const timespec_t now = sim_get_time();
        queue_->schedule_event(now.time_val + delta.time_val, fun_ptr,
                               fun_arg);
    }

    // AstraNetworkAPI pure-virtual surface: the stub has no link-bandwidth
    // semantics, a finite placeholder satisfies it.
    double get_BW_at_dimension(int /*dim*/) override {
        return 1.0;
    }

    timespec_t sim_get_time() override {
        timespec_t now;
        now.time_res = NS;
        now.time_val = queue_->get_current_time();
        return now;
    }

  private:
    std::shared_ptr<EventQueue> queue_;
    bool complete_network_callbacks_ = false;
};

const char* kDieNodeJson = R"({
  "schema_version": 1,
  "profile": "mesh_die_v1",
  "die": {
    "sram-mb-per-core": 0.75,
    "core-type": "large",
    "compute-area-mm2": 800,
    "hbm-per-side": 4,
    "hbm-sides": 1,
    "noc-width-bits": 512
  }
})";

void write_text(const std::string& path, const std::string& content) {
    std::ofstream out(path);
    if (!out.is_open()) {
        std::fprintf(stderr,
                     "[die_execution_lifecycle_test] cannot write %s\n",
                     path.c_str());
        std::exit(EXIT_FAILURE);
    }
    out << content;
}

// Norm spec; layer_count lets the caller scale the plan for long runs.
nlohmann::json spec_json(uint64_t layers) {
    nlohmann::json span;
    span["request_id"] = "r1";
    span["q_tokens"] = 8u;
    span["kv_tokens"] = 128u;
    span["q_start_pos"] = 0u;
    span["kv_start_pos"] = 0u;
    nlohmann::json spec;
    spec["schema_version"] = 1u;
    spec["op_kind"] = "norm";
    spec["phase"] = "prefill";
    spec["dtype"] = "bf16";
    spec["model_dims"] = {{"hidden", 4096u},
                          {"ffn", 11008u},
                          {"vocab", 32000u},
                          {"mlp_variant", "swiglu"},
                          {"norm_type", "rmsnorm"},
                          {"position_encoding", "rope"},
                          {"num_q_heads", 32u},
                          {"num_kv_heads", 32u},
                          {"head_dim", 128u}};
    spec["tp_slice"] = {{"tensor_parallel", 1u},
                        {"tensor_parallel_rank", 0u},
                        {"q_heads_local", 32u},
                        {"kv_heads_local", 32u},
                        {"ffn_local", 11008u},
                        {"vocab_local", 32000u}};
    spec["spans"] = nlohmann::json::array({span});
    spec["layer_count"] = layers;
    spec["weight_passes"] = 1u;
    spec["kv_access"] = {{"identity", "local"}};
    spec["fusion_policy"] = "auto";
    return spec;
}

std::shared_ptr<const DieModel::DieOpSpec> parse_spec(uint64_t layers) {
    return DieModel::parse_die_op_spec(spec_json(layers));
}

OnlineNode make_die_node(uint64_t id, uint64_t layers) {
    OnlineNode node;
    node.global_id = id;
    node.rank = 0;
    node.kind = NodeKind::Compute;
    node.node_type = 4;  // ChakraProtoMsg::COMP_NODE
    node.name = "norm_node";
    node.request_id = "r1";
    node.stage = "prefill";
    node.generation = 0;
    node.compute.die_spec = parse_spec(layers);
    return node;
}

// Independent full-speed oracle for the same plan: the exact drive policy
// DieExecution uses with hbm-bandwidth-contention=0 (Timer -> advance,
// Memory -> advance(max(1, ceil(service * 1e9 / B))) + complete).
uint64_t oracle_full_speed_elapsed(const std::shared_ptr<const DieModel::DieOpSpec>& spec,
                                   const DieModel::Hardware& hw) {
    auto plan = DieModel::compile(*spec, hw);
    DieModel::Execution machine(plan);
    while (!machine.finished()) {
        const DieModel::ExecutionAction action = machine.next_action();
        if (action.kind == DieModel::ExecutionAction::Kind::Timer) {
            machine.advance_time(std::max<uint64_t>(action.ns, 1));
        } else if (action.kind ==
                   DieModel::ExecutionAction::Kind::MemoryRequest) {
            const double ns_double =
                std::ceil(action.service_units * 1e9 / hw.hbm_bandwidth_bytes);
            machine.advance_time(std::max<uint64_t>(
                static_cast<uint64_t>(ns_double), 1));
            machine.complete_memory(action.stage_id);
        } else {
            break;
        }
    }
    return machine.elapsed_ns();
}

struct Fixture {
    std::shared_ptr<EventQueue> event_queue;
    std::unique_ptr<FakeQueueNetworkApi> network_api;
    std::unique_ptr<Sys> sys;
    NodeStoreGraphSource* source = nullptr;
};

Fixture make_fixture(const std::string& config_dir,
                     const std::string& system_file) {
    Fixture fixture;
    fixture.event_queue = std::make_shared<EventQueue>();
    fixture.network_api =
        std::make_unique<FakeQueueNetworkApi>(fixture.event_queue);
    auto source = std::make_shared<NodeStoreGraphSource>();
    fixture.source = source.get();
    fixture.sys = std::make_unique<Sys>(
        0, config_dir + "/workload", config_dir + "/comm_group.json",
        config_dir + "/" + system_file, fixture.network_api.get(),
        std::vector<int>{1}, std::vector<int>{1}, 1.0, 1.0, false,
        ExecutionMode::Online, source);
    return fixture;
}

void drain_queue(const std::shared_ptr<EventQueue>& queue) {
    while (!queue->finished()) {
        queue->proceed();
    }
}

std::vector<nlohmann::json> capture_metrics(Sys* sys, uint64_t end_tick) {
    FILE* file = std::tmpfile();
    if (file == nullptr) throw std::runtime_error("cannot capture metrics");
    const int saved = dup(STDOUT_FILENO);
    std::fflush(stdout);
    if (saved < 0 || dup2(fileno(file), STDOUT_FILENO) < 0)
        throw std::runtime_error("cannot redirect metrics");
    try {
        MetricCollector::instance().finalize({sys}, end_tick);
    } catch (...) {
        dup2(saved, STDOUT_FILENO);
        close(saved);
        std::fclose(file);
        throw;
    }
    std::fflush(stdout);
    dup2(saved, STDOUT_FILENO);
    close(saved);
    std::rewind(file);
    std::string text;
    char buffer[4096];
    while (const auto n = std::fread(buffer, 1, sizeof(buffer), file))
        text.append(buffer, n);
    std::fclose(file);
    std::vector<nlohmann::json> records;
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line))
        if (line.rfind("[METRIC] ", 0) == 0)
            records.push_back(nlohmann::json::parse(line.substr(9)));
    return records;
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    LoggerFactory::init("empty", "off");
    // Keep diagnostics visible while the metrics fixture captures stdout.
    for (const auto* name : {"statistics", "system", "HardwareResource"}) {
        LoggerFactory::get_logger(name)->sinks().push_back(
            std::make_shared<spdlog::sinks::stderr_color_sink_mt>());
    }
    CompletionObserver::instance().set_hook(tick_hook, nullptr);

    const std::string tmp_template = "/tmp/die_exec_test_XXXXXX";
    std::vector<char> buffer(tmp_template.begin(), tmp_template.end());
    buffer.push_back('\0');
    const char* const made = mkdtemp(buffer.data());
    if (made == nullptr) {
        std::fprintf(stderr,
                     "[die_execution_lifecycle_test] mkdtemp failed\n");
        return 1;
    }
    const std::string config_dir(made);
    write_text(config_dir + "/comm_group.json", "{}");
    write_text(config_dir + "/resolved_die_model.json", kDieNodeJson);
    write_text(config_dir + "/die_no_contention.json",
               R"({
  "compute-cost-model": "die",
  "hbm-bandwidth-contention": 0,
  "local-mem-latency": 100
})");
    write_text(config_dir + "/die_no_contention_no_latency.json",
               R"({
  "compute-cost-model": "die",
  "hbm-bandwidth-contention": 0
})");
    write_text(config_dir + "/die_contention.json",
               R"({
  "compute-cost-model": "die",
  "local-mem-latency": 100
})");

    const DieModel::Hardware hw =
        DieModel::make_hardware(nlohmann::json::parse(kDieNodeJson));
    const auto one_layer_spec = parse_spec(1);
    const uint64_t oracle_one_layer =
        oracle_full_speed_elapsed(one_layer_spec, hw);
    const auto reference_plan = DieModel::compile(*one_layer_spec, hw);
    const DieModel::CostResult reference =
        DieModel::estimate_alone(*reference_plan);

    // ---- Case A: contention off, latency 100, norm plan -----------------
    {
        Fixture fixture = make_fixture(config_dir, "die_no_contention.json");
        Workload& workload = *fixture.sys->workload;
        fixture.source->store().add_node(make_die_node(21, 1));
        workload.issue_dep_free_nodes();
        expect_true(workload.hw_resource->num_in_flight_gpu_comp_ops == 1,
                    "A: GPU comp slot occupied at issue");
        drain_queue(fixture.event_queue);
        const uint64_t span = g_first_tick.at(21);  // issued at tick 0
        expect_true(g_terminal_fires.at(21) == 1,
                    "A: node terminal fires exactly once");
        expect_true(span >= reference.total_ns_suggested + 100,
                    "A: span >= isolated estimate + one-shot latency");
        expect_true(span == oracle_one_layer + 100,
                    "A: span == oracle machine elapsed + exactly one 100 ns "
                    "local_mem_latency charge");
        expect_true(workload.hw_resource->num_in_flight_gpu_comp_ops == 0,
                    "A: GPU comp slot released after Done");
        expect_true(
            !workload.die_model_adapter()->has_active_die_executions(),
            "A: adapter drains after Done");
    }

    // ---- Case B: contention off, no latency ------------------------------
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        Fixture fixture =
            make_fixture(config_dir, "die_no_contention_no_latency.json");
        Workload& workload = *fixture.sys->workload;
        fixture.source->store().add_node(make_die_node(22, 1));
        workload.issue_dep_free_nodes();
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires.at(22) == 1,
                    "B: node terminal fires exactly once");
        expect_true(g_first_tick.at(22) == oracle_one_layer,
                    "B: span == oracle machine elapsed exactly (latency 0 "
                    "-> no latency charge)");
    }

    // ---- Case C: contention on ------------------------------------------
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        Fixture fixture = make_fixture(config_dir, "die_contention.json");
        Workload& workload = *fixture.sys->workload;
        fixture.source->store().add_node(make_die_node(23, 1));
        workload.issue_dep_free_nodes();
        // Mid-flight: pump a few events, the norm plan is far from done.
        for (int i = 0; i < 5 && !fixture.event_queue->finished(); ++i) {
            fixture.event_queue->proceed();
        }
        expect_true(
            workload.die_model_adapter()->has_active_die_executions(),
            "C: adapter reports an active execution mid-flight");
        expect_true(workload.hw_resource->num_in_flight_gpu_comp_ops == 1,
                    "C: GPU comp slot still occupied mid-flight");
        drain_queue(fixture.event_queue);
        const uint64_t span = g_first_tick.at(23);
        expect_true(g_terminal_fires.at(23) == 1,
                    "C: node terminal fires exactly once under contention");
        expect_true(span >= reference.total_ns_suggested + 100,
                    "C: span >= isolated estimate + latency (equal share "
                    "can only delay)");
        expect_true(
            !workload.die_model_adapter()->has_active_die_executions(),
            "C: adapter drains after Done under contention");
        expect_true(
            workload.local_hbm_bandwidth_model != nullptr &&
                workload.local_hbm_bandwidth_model
                        ->die_service_units_served() > 0.0 &&
                workload.local_hbm_bandwidth_model
                        ->die_actual_bytes_served() > 0.0,
            "C: die side-band records service AND actual bytes (separate "
            "columns)");
    }

    // ---- Case D: marker micro-job, 1 ns event arm ------------------------
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        Fixture fixture = make_fixture(config_dir, "die_no_contention.json");
        Workload& workload = *fixture.sys->workload;
        OnlineNode marker;
        marker.global_id = 24;
        marker.rank = 0;
        marker.kind = NodeKind::Compute;
        marker.node_type = 4;
        marker.name = "first_token";
        marker.request_id = "r1";
        marker.stage = "prefill";
        marker.generation = 0;
        marker.compute.op_role = OpRole::ControlMarker;
        marker.compute.num_ops = 1;
        marker.compute.tensor_size = 1;
        fixture.source->store().add_node(marker);
        workload.issue_dep_free_nodes();
        const uint64_t issue_tick = Sys::boostedTick();  // == 0
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires.at(24) == 1 &&
                    g_first_tick.at(24) == issue_tick + 1,
                    "D: marker without the HBM model completes on the 1 ns "
                    "event (legacy micro-job)");
    }

    // ---- Case E: teardown of an in-flight execution ----------------------
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        Fixture fixture = make_fixture(config_dir, "die_contention.json");
        Workload& workload = *fixture.sys->workload;
        fixture.source->store().add_node(make_die_node(25, 8));  // big plan
        workload.issue_dep_free_nodes();
        for (int i = 0; i < 8 && !fixture.event_queue->finished(); ++i) {
            fixture.event_queue->proceed();
        }
        expect_true(g_terminal_fires.count(25) == 0,
                    "E: big node still in flight after partial pumping");
        // Tear the whole Sys down mid-flight (Workload destructor ->
        // adapter shutdown -> every DieExecution cancels its pending event
        // and drops its pending HBM die job). No completion may fire.
        fixture.sys.reset();
        // Stale outer alarms (the legacy fake backend cannot cancel
        // physically) must drain through the retired-Sys guard harmlessly.
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires.count(25) == 0,
                    "E: teardown fires no completion for the in-flight "
                    "node (exactly-zero terminals)");
    }

    // F: typed COMP and a real COMM_SEND share one coarse Runtime. The
    // fake network never completes; cancellation/teardown must retain the
    // delivered prefix and must not publish an outer node completion.
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        std::filesystem::path repo(__FILE__);
        for (unsigned i = 0; i < 5; ++i) repo = repo.parent_path();
        std::ifstream hardware_file(repo / "extern/die_model/coarse/configs/example_hardware.json");
        nlohmann::json hardware;
        hardware_file >> hardware;
        write_text(config_dir + "/coarse_hardware.json", hardware.dump());
        write_text(config_dir + "/coarse_system.json", nlohmann::json{
            {"compute-cost-model", "die"}, {"die-model-profile", "coarse_die_v2"},
            {"die-model-config", config_dir + "/coarse_hardware.json"},
            {"hbm-bandwidth-contention", 1}}.dump());
        Fixture fixture = make_fixture(config_dir, "coarse_system.json");
        // Observe the cancellation prefix with the existing microbenchmark
        // history policy; service finalization still rejects live GPU nodes.
        fixture.sys->workload->stats->configure_online_history_preservation(true);
        auto& bridge = *fixture.sys->coarse_die_runtime;
        nlohmann::json task{{"op_kind", "gemm"},
            {"tensor_shapes", {{"m", 128}, {"n", 128}, {"k", 128}}},
            {"storage_dtype", "bf16"}};
        const auto plan = bridge.compile(task);
        OnlineNode comp;
        comp.global_id = 40; comp.rank = 0; comp.kind = NodeKind::Compute;
        comp.node_type = 4; comp.name = "coarse_comp";
        comp.request_id = "r1"; comp.stage = "prefill";
        comp.compute.die_task_json = std::make_shared<const nlohmann::json>(task);
        fixture.source->store().add_node(comp);
        OnlineNode comm;
        comm.global_id = 41; comm.rank = 0; comm.kind = NodeKind::CommSend;
        comm.node_type = 5; comm.name = "coarse_send";
        comm.request_id = "r1"; comm.stage = "prefill";
        comm.comm.bytes = 1ULL << 30; comm.comm.src = 0; comm.comm.dst = 0;
        comm.comm.tag = 0; comm.comm.hbm_charge = true;
        fixture.source->store().add_node(comm);
        fixture.sys->workload->issue_dep_free_nodes();
        nlohmann::json partial;
        for (unsigned i = 0; i < 32 && !fixture.event_queue->finished(); ++i) {
            fixture.event_queue->proceed();
            partial = bridge.counters_at(Sys::boostedTick());
            const double ops = partial["served_by_category"]["comp"]["int4_ops"].get<double>();
            if (ops > 0 && ops < plan->summary["executed_int4_ops"].get<double>()) break;
        }
        const double delivered = partial["served_by_category"]["comp"]["int4_ops"].get<double>();
        expect_true(delivered > 0 && delivered < plan->summary["executed_int4_ops"].get<double>(),
                    "F: real typed COMP has positive partial INT4 service");
        expect_true(partial["served_by_category"]["comm_send"]["hbm_served_bytes_by_row_direction"][0][0].get<double>() > 0 &&
                    partial["hbm_shared_ns"].get<double>() > 0,
                    "F: COMP and declared COMM source read share local HBM");
        fixture.sys->workload->die_model_adapter()->shutdown();
        const auto cancelled = bridge.counters_at(Sys::boostedTick());
        expect_true(cancelled["served_by_category"]["comp"]["int4_ops"].get<double>() == delivered &&
                    g_terminal_fires.count(40) == 0 && g_terminal_fires.count(41) == 0,
                    "F: teardown cancellation preserves delivered work without graph terminals");
        write_text(config_dir + "/coarse_manifest.json", nlohmann::json{
            {"schema_version", 1}, {"repo_variant", "coarse-lifecycle-fixture"},
            {"run_mode", "microbenchmark"}, {"run_id", "coarse-lifecycle-fixture"},
            {"requests", nlohmann::json::array()},
            {"node_events_by_rank", nlohmann::json::object()},
            {"planner_memory_peaks", nlohmann::json::array()},
            {"memory_actions", nlohmann::json::array()}}.dump());
        MetricCollector::instance().initialize(config_dir + "/coarse_manifest.json", "summary");
        const auto records = capture_metrics(fixture.sys.get(), Sys::boostedTick());
        nlohmann::json rank_record, hbm_record;
        for (const auto& record : records) {
            if (record.value("type", std::string()) == "rank_compute") rank_record = record;
            if (record.value("type", std::string()) == "local_hbm") hbm_record = record;
        }
        expect_true(!rank_record.is_null() && !hbm_record.is_null(),
                    "F: MetricCollector emits coarse rank and HBM records");
        if (!rank_record.is_null() && !hbm_record.is_null()) {
            const auto& ledger = rank_record["coarse_die_runtime"];
            expect_true(ledger["served_by_category"] == cancelled["served_by_category"] &&
                        ledger["accounting_history_complete"] == true,
                        "F: rank metrics retain cancelled partial typed service");
            const auto& rows = cancelled["hbm_served_bytes_by_row_direction"];
            const double total = rows[0][0].get<double>() + rows[0][1].get<double>() +
                                 rows[1][0].get<double>() + rows[1][1].get<double>();
            const auto& raw = cancelled["hardware_raw"]["hbm_row_peak_Bps"];
            const double denominator = (raw[0].get<double>() + raw[1].get<double>()) *
                                       (static_cast<double>(Sys::boostedTick()) / 1e9);
            expect_true(std::abs(hbm_record["dram_bw_util_numerator_bytes"].get<double>() - total) < 1e-6 &&
                        std::abs(hbm_record["dram_bw_util"].get<double>() - total / denominator) < 1e-12 &&
                        hbm_record["compute_bytes_served"].get<double>() == 0,
                        "F: total HBM numerator includes DIE and COMM once over raw peaks");
        }
        fixture.sys.reset();
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires.count(40) == 0 && g_terminal_fires.count(41) == 0,
                    "F: COMP plus COMM teardown drains stale alarms without callbacks");
    }

    // G: the normal service path joins a completed network callback with
    // local HBM, retires both nodes once, and finalizes the lifetime ledger.
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        Fixture fixture = make_fixture(config_dir, "coarse_system.json");
        fixture.network_api->complete_network_callbacks(true);
        auto& bridge = *fixture.sys->coarse_die_runtime;
        nlohmann::json task{{"op_kind", "gemm"},
            {"tensor_shapes", {{"m", 128}, {"n", 128}, {"k", 128}}},
            {"storage_dtype", "bf16"}};
        const auto plan = bridge.compile(task);
        OnlineNode comp;
        comp.global_id = 50; comp.rank = 0; comp.kind = NodeKind::Compute;
        comp.node_type = 4; comp.name = "coarse_complete";
        comp.request_id = "r1"; comp.stage = "prefill";
        comp.compute.die_task_json = std::make_shared<const nlohmann::json>(task);
        fixture.source->store().add_node(comp);
        OnlineNode comm;
        comm.global_id = 51; comm.rank = 0; comm.kind = NodeKind::CommSend;
        comm.node_type = 5; comm.name = "coarse_complete_send";
        comm.request_id = "r1"; comm.stage = "prefill";
        comm.comm.bytes = 1ULL << 20; comm.comm.src = 0; comm.comm.dst = 0;
        comm.comm.tag = 0; comm.comm.hbm_charge = true;
        fixture.source->store().add_node(comm);
        fixture.sys->workload->issue_dep_free_nodes();
        drain_queue(fixture.event_queue);
        const auto completed = bridge.counters_at(Sys::boostedTick());
        expect_true(g_terminal_fires[50] == 1 && g_terminal_fires[51] == 1 &&
                    fixture.sys->workload->hw_resource->num_in_flight_gpu_comp_ops == 0 &&
                    !fixture.sys->workload->die_model_adapter()->has_active_die_executions(),
                    "G: typed COMP and COMM join complete once and release the GPU");
        const auto& categories = completed["served_by_category"];
        expect_true(categories["comp"]["int4_ops"].get<double>() ==
                        plan->summary["executed_int4_ops"].get<double>() &&
                    categories["comm_send"]["hbm_served_bytes_by_row_direction"][0][0].get<double>() +
                        categories["comm_send"]["hbm_served_bytes_by_row_direction"][1][0].get<double>() ==
                        static_cast<double>(comm.comm.bytes),
                    "G: retired COMP and COMM conserve their typed served work");
        nlohmann::json manifest;
        std::ifstream(config_dir + "/coarse_manifest.json") >> manifest;
        manifest["run_mode"] = "service";
        write_text(config_dir + "/coarse_manifest.json", manifest.dump());
        MetricCollector::instance().initialize(config_dir + "/coarse_manifest.json", "summary");
        const auto records = capture_metrics(fixture.sys.get(), Sys::boostedTick());
        bool matched = false;
        for (const auto& record : records)
            if (record.value("type", std::string()) == "rank_compute")
                matched = record["coarse_die_runtime"]["served_by_category"] == categories;
        expect_true(matched, "G: normal service finalization retains retired coarse counters");
        fixture.sys.reset();
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires[50] == 1 && g_terminal_fires[51] == 1,
                    "G: completed teardown publishes no duplicate terminals");
    }

    std::error_code ec;
    std::filesystem::remove_all(config_dir, ec);
    if (!g_ok) {
        std::fprintf(stderr,
                     "[die_execution_lifecycle_test] FAIL (see above)\n");
        return 1;
    }
    std::printf("[die_execution_lifecycle_test] ALL PASS\n");
    return 0;
}
