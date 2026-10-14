/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root of this source tree.
*******************************************************************************/

// Coarse workspace/runtime seam: typed COMP Plans share the rank Runtime
// with endpoint transfers, receive capacity before admission, and release the
// exact lease on completion or cancellation.

#include <astra-network-analytical/common/EventQueue.h>

#include <coarse_die/Plan.hh>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/CoarseDieRuntime.hh"
#include "astra-sim/workload/LocalHbmBandwidthModel.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "astra-sim/workload/WorkspaceAdmission.hh"

using namespace AstraSim;
using namespace AstraSim::ExecutionDriven;
using namespace NetworkAnalytical;
using json = nlohmann::json;

namespace {

bool g_ok = true;
uint64_t g_terminal_count = 0;

void expect(bool condition, const char* message) {
    std::printf("%-78s %s\n", message, condition ? "PASS" : "FAIL");
    if (!condition) g_ok = false;
}

void terminal_hook(void*, int, uint64_t, const char*, const char*, uint64_t,
                   uint64_t, int) {
    ++g_terminal_count;
}

class TestRemoteMemoryApi final : public AstraRemoteMemoryAPI {
  public:
    void set_sys(int, Sys*) override {}
    void issue(uint64_t, WorkloadLayerHandlerData*) override {}
};

class FakeQueueNetworkApi final : public AstraNetworkAPI {
  public:
    explicit FakeQueueNetworkApi(std::shared_ptr<EventQueue> queue)
        : AstraNetworkAPI(0), queue_(std::move(queue)) {}

    int sim_send(void*, uint64_t, int, int, int, sim_request*,
                 void (*)(void*), void*) override {
        return 0;
    }
    int sim_recv(void*, uint64_t, int, int, int, sim_request*,
                 void (*)(void*), void*) override {
        return 0;
    }
    void sim_schedule(timespec_t delta, void (*callback)(void*),
                      void* argument) override {
        const timespec_t now = sim_get_time();
        queue_->schedule_event(now.time_val + delta.time_val,
                               callback, argument);
    }
    timespec_t sim_get_time() override {
        timespec_t now;
        now.time_res = NS;
        now.time_val = queue_->get_current_time();
        return now;
    }

  private:
    std::shared_ptr<EventQueue> queue_;
};

struct Fixture {
    std::shared_ptr<EventQueue> event_queue = std::make_shared<EventQueue>();
    std::unique_ptr<FakeQueueNetworkApi> network_api;
    TestRemoteMemoryApi remote_memory;
    std::shared_ptr<NodeStoreGraphSource> source =
        std::make_shared<NodeStoreGraphSource>();
    std::unique_ptr<Sys> sys;

    Fixture(const std::string& config_dir, const std::string& system_file) {
        network_api = std::make_unique<FakeQueueNetworkApi>(event_queue);
        sys = std::make_unique<Sys>(
            0, config_dir + "/workload", config_dir + "/comm_group.json",
            config_dir + "/" + system_file, &remote_memory,
            network_api.get(), std::vector<int>{1}, std::vector<int>{1},
            1.0, 1.0, false, ExecutionMode::Online, source);
    }
};

void write_text(const std::string& path, const std::string& content) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write test input " + path);
    out << content;
}

json task_json() {
    return {
        {"schema", "coarse_die_task_v2"},
        {"op_kind", "attention_region"},
        {"phase", "prefill"},
        {"tensor_shapes", {{"m", 64}, {"q_heads", 1},
                            {"kv_heads", 1}, {"head_dim", 32}}},
        {"spans", json::array({{{"q_tokens", 64}, {"kv_tokens", 64},
                                 {"q_start_pos", 0}, {"kv_start_pos", 0}}})},
        // Force the materialized mapping so the selected typed Plan has a
        // concrete HBM scratch reservation for the admission seam.
        {"fusion_policy", "materialized"},
    };
}

OnlineNode coarse_node(const uint64_t id, const json& task) {
    OnlineNode node;
    node.global_id = id;
    node.rank = 0;
    node.kind = NodeKind::Compute;
    node.node_type = 4;  // ChakraProtoMsg::COMP_NODE
    node.name = "typed_coarse_comp";
    node.request_id = "coarse-request";
    node.stage = "prefill";
    node.generation = 0;
    node.compute.op_role = OpRole::Compute;
    node.compute.runtime_ns = 0;
    node.compute.die_task_json =
        std::make_shared<const json>(task);
    return node;
}

void install_capacity(WorkspaceAdmission& ledger, uint64_t capacity,
                      uint64_t revision = 0) {
    WorkspaceCapacityRank snapshot;
    snapshot.rank = ledger.rank();
    snapshot.workspace_revision = revision;
    snapshot.capacity_bytes = capacity;
    snapshot.model_weight_bytes = 100;
    ledger.validate_capacity_snapshot(snapshot);
    ledger.install_capacity_snapshot_unchecked(snapshot);
}

void test_acquire_block_retry_shared_endpoint_and_release(
    const std::string& config_dir) {
    Fixture fixture(config_dir, "coarse_system.json");
    expect(fixture.sys->coarse_die_runtime != nullptr &&
               fixture.sys->workspace_admission != nullptr &&
               fixture.sys->die_runtime == nullptr,
           "coarse profile creates one independent rank Runtime and ledger");
    if (fixture.sys->coarse_die_runtime == nullptr ||
        fixture.sys->workspace_admission == nullptr) return;

    const json task = task_json();
    const auto plan = fixture.sys->coarse_die_runtime->compile(task);
    const uint64_t peak = coarse_die::hbm_workspace_peak_bytes(*plan);
    expect(peak > 0, "typed attention Plan has an explicit workspace peak");
    if (peak == 0) return;
    const uint64_t capacity = 100 + peak;
    install_capacity(*fixture.sys->workspace_admission, capacity);

    auto held = fixture.sys->workspace_admission->try_acquire(999, 1, peak);
    expect(held.status == WorkspaceAdmission::AcquireStatus::Acquired &&
               fixture.sys->workspace_admission->active_workspace_bytes() == peak,
           "workspace lease is acquired against the installed scalar capacity");

    fixture.source->store().add_node(coarse_node(41, task));
    fixture.sys->workload->issue_dep_free_nodes();
    expect(fixture.sys->workload->hw_resource->num_in_flight_gpu_comp_ops == 0 &&
               fixture.source->store().resolve_free_nodes() ==
                   std::vector<uint64_t>({41}),
           "a competing ready Plan stays free and unoccupied while capacity is held");

    held.lease.release();
    expect(fixture.sys->workspace_admission->active_workspace_bytes() == 0,
           "explicit cancellation/release returns the exact byte hold");
    fixture.sys->workload->issue_dep_free_nodes();
    expect(fixture.sys->workspace_admission->active_lease_count() == 1 &&
               fixture.sys->workspace_admission->active_workspace_bytes() == peak &&
               fixture.sys->workload->hw_resource->num_in_flight_gpu_comp_ops == 1,
           "the same typed Plan acquires workspace before COMP issue");

    bool endpoint_done = false;
    fixture.sys->workload->local_hbm_bandwidth_model->issue_transfer(
        LocalHbmBandwidthModel::JobKind::COMM_READ, 1000000000ULL, 0, 900,
        [&endpoint_done] { endpoint_done = true; });
    expect(fixture.sys->coarse_die_runtime->has_active_executions() &&
               fixture.sys->workload->die_model_adapter()
                   ->has_active_die_executions(),
           "Plan and endpoint transfer are submitted to the shared Runtime");

    std::size_t event_steps = 0;
    while (fixture.sys->workspace_admission->active_lease_count() != 0 &&
           !fixture.event_queue->finished() && event_steps < 100000) {
        fixture.event_queue->proceed();
        ++event_steps;
    }
    expect(fixture.sys->workspace_admission->active_lease_count() == 0 &&
               fixture.sys->workspace_admission->active_workspace_bytes() == 0,
           "normal typed Plan completion releases its one workspace lease");
    expect(!endpoint_done &&
               fixture.sys->coarse_die_runtime->has_active_executions(),
           "long endpoint remains active after the COMP lease releases");

    while (!fixture.event_queue->finished() && event_steps < 200000) {
        fixture.event_queue->proceed();
        ++event_steps;
    }
    expect(endpoint_done &&
               !fixture.sys->coarse_die_runtime->has_active_executions() &&
               g_terminal_count == 1,
           "shared Runtime drains endpoint and emits exactly one COMP terminal");
}

void test_adapter_cancellation_releases_lease_without_terminal(
    const std::string& config_dir) {
    Fixture fixture(config_dir, "coarse_system.json");
    const json task = task_json();
    const auto plan = fixture.sys->coarse_die_runtime->compile(task);
    const uint64_t peak = coarse_die::hbm_workspace_peak_bytes(*plan);
    install_capacity(*fixture.sys->workspace_admission, 100 + peak);
    fixture.source->store().add_node(coarse_node(42, task));
    fixture.sys->workload->issue_dep_free_nodes();
    const uint64_t terminals_before = g_terminal_count;
    expect(fixture.sys->workspace_admission->active_lease_count() == 1 &&
               fixture.sys->coarse_die_runtime->has_active_executions(),
           "cancellation fixture starts with a live Plan and lease");

    fixture.sys->workload->die_model_adapter()->shutdown();
    expect(fixture.sys->workspace_admission->active_lease_count() == 0 &&
               fixture.sys->workspace_admission->active_workspace_bytes() == 0 &&
               !fixture.sys->coarse_die_runtime->has_active_executions(),
           "adapter cancellation destroys Runtime execution and releases lease");
    expect(g_terminal_count == terminals_before,
           "cancellation never synthesizes an outer COMP completion");
}

}  // namespace

int main() {
    const std::string tmp_template = "/tmp/coarse_workspace_runtime_XXXXXX";
    std::vector<char> directory(tmp_template.begin(), tmp_template.end());
    directory.push_back('\0');
    const char* made = mkdtemp(directory.data());
    if (made == nullptr) {
        std::fprintf(stderr, "[coarse_workspace_runtime_test] mkdtemp failed\n");
        return 1;
    }
    const std::string config_dir(made);
    try {
        std::filesystem::copy_file(
            std::string(COARSE_DIE_SOURCE_DIR) + "/configs/example_hardware.json",
            config_dir + "/coarse_hardware.json");
        write_text(config_dir + "/comm_group.json", "{}");
        write_text(config_dir + "/coarse_system.json",
            std::string(R"({
  "scheduling-policy": "FIFO",
  "compute-cost-model": "die",
  "die-model-profile": "coarse_die_v2",
  "die-model-config": ")") + config_dir + R"(/coarse_hardware.json",
  "local-mem-latency": 20
})");
        CompletionObserver::instance().set_hook(terminal_hook, nullptr);
        test_acquire_block_retry_shared_endpoint_and_release(config_dir);
        test_adapter_cancellation_releases_lease_without_terminal(config_dir);
        CompletionObserver::instance().set_hook(nullptr, nullptr);
    } catch (const std::exception& error) {
        std::fprintf(stderr,
                     "[coarse_workspace_runtime_test] exception: %s\n",
                     error.what());
        g_ok = false;
    }
    std::filesystem::remove_all(config_dir);
    if (g_ok) {
        std::printf("[coarse_workspace_runtime_test] ALL PASS\n");
        return 0;
    }
    return 1;
}
