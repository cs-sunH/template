// Integration-only probe. Reuse each repository's actual Sys/Graph fixture,
// without editing production tasks or running the fixture's renamed main.
#define main availability_original_fixture_main
#include CONSUMER_FIXTURE_SOURCE
#undef main

namespace {
uint64_t probe_terminals = 0;
void observe_probe_terminal(void*, int, uint64_t, const char*, const char*,
                            uint64_t, uint64_t, int) { ++probe_terminals; }
void check_probe(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(reason);
    std::printf("repeat-prepare: %s PASS\n", reason);
}
auto probe_fixture(const std::string& config) {
#ifdef AVAILABILITY_JOINT
    return Fixture(config, "coarse_system.json");
#else
    return make_fixture(config, "coarse_system.json");
#endif
}
auto probe_node(Fixture& fixture, const nlohmann::json& task) {
    OnlineNode node;
    node.global_id = 975; node.rank = 0; node.generation = 7;
    node.kind = NodeKind::Compute; node.node_type = 4;
    node.name = "actual_repeated_region"; node.request_id = "availability";
    node.stage = "prefill"; node.compute.op_role = OpRole::Compute;
    node.compute.die_task_json = std::make_shared<const nlohmann::json>(task);
    fixture.source->store().add_node(node);
#ifdef AVAILABILITY_JOINT
    return *fixture.source->lookup(node.global_id);
#else
    return *fixture.source->lookup_ptr(node.global_id);
#endif
}
nlohmann::json observe_probe(Fixture& fixture) {
    std::vector<uint64_t> ready;
    fixture.source->store().fill_free_node_snapshot(ready);
    auto& ledger = *fixture.sys->workspace_admission;
    auto& bridge = *fixture.sys->coarse_die_runtime;
    return {{"runtime", bridge.counters()}, {"cache", bridge.plan_cache_counters()},
        {"tick", Sys::boostedTick()}, {"ready", ready},
        {"lease_count", ledger.active_lease_count()},
        {"workspace_bytes", ledger.active_workspace_bytes()},
        {"revision", ledger.workspace_revision()},
        {"gpu", fixture.sys->workload->hw_resource->num_in_flight_gpu_comp_ops},
        {"terminals", probe_terminals}};
}
template<class F> void must_reject_probe(F action, const char* reason) {
    bool rejected = false;
    try { action(); } catch (const std::exception& error) {
        std::printf("repeat-prepare rejection: %s\n", error.what());
        rejected = std::string(error.what()).find("prediction_not_available:") != std::string::npos;
    }
    check_probe(rejected, reason);
}
}

int main() {
    std::filesystem::path config;
    try {
        char directory[] = "/tmp/availability_repeat_XXXXXX";
        const char* made = mkdtemp(directory);
        check_probe(made != nullptr, "temporary fixture directory");
        config = made;
        write_text((config / "comm_group.json").string(), "{}");
        std::filesystem::copy_file(std::string(AVAILABILITY_SDK_ROOT) +
            "/configs/example_hardware.json", config / "coarse_hardware.json");
        write_text((config / "coarse_system.json").string(), nlohmann::json{
            {"scheduling-policy", "FIFO"}, {"compute-cost-model", "die"},
            {"die-model-profile", "coarse_die_v2"},
            {"die-model-config", (config / "coarse_hardware.json").string()},
            {"local-mem-latency", 100}}.dump());
        CompletionObserver::instance().set_hook(observe_probe_terminal, nullptr);
        nlohmann::json task{{"schema", "coarse_die_task_v2"}, {"op_kind", "ffn_region"},
            {"phase", "prefill"}, {"tensor_shapes", {{"m", 64}, {"hidden", 128},
                {"ffn", 256}, {"mlp_variant", "swiglu"}}},
            {"fusion_policy", "materialized"}, {"storage_dtype", "bf16"},
            {"layer_instances", 1}, {"weight_passes", 1}};
        {
            auto fixture = probe_fixture(config.string());
            const auto node = probe_node(fixture, task);
            auto& adapter = *fixture.sys->workload->die_model_adapter();
            const auto first = adapter.prepare_coarse_plan(node);
            const auto before = observe_probe(fixture);
            const auto second = adapter.prepare_coarse_plan(node);
            check_probe(first.plan == second.plan && first.workspace_bytes > 0 &&
                *first.task == task && before == observe_probe(fixture),
                "real R1 cold/hot preparation remains valid and cache-stable");
        }
        task["layer_instances"] = 3;
        auto fixture = probe_fixture(config.string());
        auto& bridge = *fixture.sys->coarse_die_runtime;
        auto& workload = *fixture.sys->workload;
        auto& adapter = *workload.die_model_adapter();
        auto& ledger = *fixture.sys->workspace_admission;
        const auto node = probe_node(fixture, task);
        const auto body = coarse_die::compile(bridge.hardware(), task);
        check_probe(body->summary.at("prediction_status") == "not_available" &&
                    body->summary.at("alone_estimate_status") == "not_available",
                    "actual new SDK R3 compiler marks incomplete body timing");
        coarse_die::Runtime diagnostic(bridge.hardware());
        const auto handle = diagnostic.submit(body, coarse_die::Owner{1, 975, 7, 1}, 0);
        diagnostic.cancel(handle, 0);
        check_probe(diagnostic.query_progress(handle).at("state") == "cancelled",
                    "public R3 body diagnostics remain usable");
        const auto peak = coarse_die::hbm_workspace_peak_bytes(*body);
        const WorkspaceCapacityRank capacity{fixture.sys->id,
            ledger.workspace_revision(), peak + 4096, 0, 0, 0};
        ledger.validate_capacity_snapshot(capacity);
        ledger.install_capacity_snapshot_unchecked(capacity);
        fixture.event_queue->schedule_event(17, [](void*) {}, nullptr);
        while (fixture.event_queue->get_current_time() < 17) fixture.event_queue->proceed();
        const auto before = observe_probe(fixture);
        check_probe(before.at("runtime").at("now_ns") == 0 && before.at("tick") == 17,
                    "R3 rejection starts with Sys ahead of untouched Runtime");
        for (int attempt = 0; attempt < 2; ++attempt) {
            must_reject_probe([&] { (void)bridge.compile(task); }, "actual R3 compile boundary rejects");
            must_reject_probe([&] { (void)adapter.prepare_coarse_plan(node); }, "actual R3 prepare boundary rejects");
            check_probe(observe_probe(fixture) == before && *node.compute.die_task_json == task,
                        "failed R3 compile/prepare leaves cache, graph, clocks, leases and payload intact");
        }
        must_reject_probe([&] { workload.issue_dep_free_nodes(); },
                          "actual R3 ready-scan issue rejects before dependency release");
        check_probe(observe_probe(fixture) == before,
                    "failed R3 issue leaves graph/GPU/runtime/lease/terminal state intact");
        std::printf("repeat-prepare evidence: %s\n", nlohmann::json{
            {"task", task}, {"summary", body->summary}, {"before", before},
            {"hardware_fingerprint", bridge.hardware().fingerprint}}.dump().c_str());
        fixture.sys.reset();
        while (!fixture.event_queue->finished()) fixture.event_queue->proceed();
        check_probe(probe_terminals == 0, "no deferred terminal after rejected R3 teardown");
        CompletionObserver::instance().set_hook(nullptr, nullptr);
        std::filesystem::remove_all(config);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "repeat-prepare failed: %s\n", error.what());
        if (!config.empty()) std::filesystem::remove_all(config);
        return 1;
    }
}
