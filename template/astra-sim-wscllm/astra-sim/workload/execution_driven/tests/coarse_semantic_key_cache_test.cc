/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/**
coarse_semantic_key_cache_test.cc -- CoarseDieRuntime semantic compile-cache
key regression (roofline batch P2-C; hermetic online fixture, no
sh_test_mesh artifacts beyond the repo's coarse example hardware).

Same fixture shape as local_hbm_die_memory_test.cc: an online Sys over a
NodeStoreGraphSource (fake network API on a real EventQueue) with a
coarse_die_v2 system file. Nothing is submitted, so the event queue stays
empty and only the compile cache is exercised.

Cases:
  A. same task twice: the first call compiles (miss), the second is a hit
     returning the same Plan instance.
  B. identity-only change (different span request ids, provenance, and
     config hash): hits the plan compiled under the other tracing identity
     instead of recompiling an identical semantic task.
  C. one semantic field change (span kv_tokens): a fresh miss producing a
     distinct cache entry and a different Plan.
  D. a second runtime whose hardware carries a different fingerprint never
     reuses the first runtime's entry (fresh miss there, hit only inside
     its own cache).
  E. cache observation counters stay machine-readable: hits/misses/entries,
     plan_cache_key_ns_total, schedule_next_calls, and the coarse Runtime
     observation fields merged into counters().

Build: cmake target
      AstraSim_Analytical_Congestion_Aware_CoarseSemanticKeyCacheTest.
Exit code 0 on ALL PASS.
*/

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/CoarseDieRuntime.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include <astra-network-analytical/common/EventQueue.h>
// The repositories with an AstraRemoteMemoryAPI backend take it as an extra
// Sys constructor argument ahead of the network API; the others construct Sys
// without it. Probe the header so this shared test builds in all five.
#if __has_include("astra-sim/common/AstraRemoteMemoryAPI.hh")
#include "astra-sim/common/AstraRemoteMemoryAPI.hh"
#define COARSE_SEMANTIC_KEY_CACHE_TEST_REMOTE_MEM 1
#endif

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
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

class FakeQueueNetworkApi final : public AstraNetworkAPI {
  public:
    explicit FakeQueueNetworkApi(std::shared_ptr<EventQueue> queue)
        : AstraNetworkAPI(0), queue_(std::move(queue)) {}

    int sim_send(void*, uint64_t, int, int, int, sim_request*,
                 void (*callback)(void*), void* data) override {
        queue_->schedule_event(queue_->get_current_time() + 1,
                               callback, data);
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

    timespec_t sim_get_time() override {
        timespec_t now;
        now.time_res = NS;
        now.time_val = queue_->get_current_time();
        return now;
    }

    // wscllm's AstraNetworkAPI adds get_BW_at_dimension as a pure virtual;
    // its own test stubs return a finite placeholder (the stub carries no
    // link-bandwidth semantics). Declared without `override` so the same
    // file also compiles where the base declares no such virtual.
    double get_BW_at_dimension(int) { return 1.0; }

  private:
    std::shared_ptr<EventQueue> queue_;
};

#ifdef COARSE_SEMANTIC_KEY_CACHE_TEST_REMOTE_MEM
// Same contract as die_execution_lifecycle_test.cc: the fixture never issues
// remote MEM nodes, so issue() is fatal and set_sys() stays a no-op.
class FakeRemoteMemoryApi final : public AstraRemoteMemoryAPI {
  public:
    void set_sys(int, Sys*) override {}
    void issue(uint64_t, WorkloadLayerHandlerData*) override {
        std::fprintf(stderr,
                     "[coarse_semantic_key_cache_test] "
                     "FakeRemoteMemoryApi::issue called -- the fixture never "
                     "issues remote MEM nodes\n");
        std::exit(EXIT_FAILURE);
    }
};
#endif

struct Fixture {
    std::shared_ptr<EventQueue> event_queue;
    std::unique_ptr<FakeQueueNetworkApi> network_api;
#ifdef COARSE_SEMANTIC_KEY_CACHE_TEST_REMOTE_MEM
    std::unique_ptr<FakeRemoteMemoryApi> remote_api;
#endif
    std::unique_ptr<Sys> sys;
};

void write_text(const std::string& path, const std::string& content) {
    std::ofstream out(path);
    if (!out.is_open()) {
        std::fprintf(stderr,
                     "[coarse_semantic_key_cache_test] cannot write %s\n",
                     path.c_str());
        std::exit(1);
    }
    out << content;
}

// Legacy DieOpSpec in the exact shape the FACE emitter produces: spans
// carry per-span request ids and the wrapper carries provenance and a
// config hash. The semantic cache key drops all three.
nlohmann::json legacy_ffn_spec(const std::string& span0_id,
                               const std::string& span1_id,
                               const std::string& provenance,
                               const std::string& config) {
    return nlohmann::json{
        {"schema_version", 1},
        {"op_kind", "ffn_region"},
        {"phase", "decode"},
        {"dtype", "bf16"},
        {"model_dims",
         {{"hidden", 64},
          {"ffn", 128},
          {"vocab", 256},
          {"mlp_variant", "gelu"},
          {"norm_type", "layernorm"},
          {"position_encoding", "none"},
          {"num_q_heads", 8},
          {"num_kv_heads", 8},
          {"head_dim", 8}}},
        {"tp_slice",
         {{"tensor_parallel", 2},
          {"tensor_parallel_rank", 0},
          {"q_heads_local", 4},
          {"kv_heads_local", 4},
          {"ffn_local", 64},
          {"vocab_local", 128}}},
        {"spans",
         nlohmann::json::array({
             nlohmann::json{{"request_id", span0_id},
                            {"q_tokens", 1},
                            {"kv_tokens", 1},
                            {"q_start_pos", 0},
                            {"kv_start_pos", 0}},
             nlohmann::json{{"request_id", span1_id},
                            {"q_tokens", 1},
                            {"kv_tokens", 2},
                            {"q_start_pos", 1},
                            {"kv_start_pos", 0}},
         })},
        {"kv_access", {{"identity", "local"}}},
        {"fusion_policy", "auto"},
        {"layer_count", 2},
        {"weight_passes", 2},
        {"repeat_seams", {{"mode", "serial_on_drain"}}},
        {"provenance", provenance},
        {"config_hash", config}};
}

Fixture make_fixture(const std::string& config_dir,
                     const std::string& system_file) {
    Fixture fixture;
    fixture.event_queue = std::make_shared<EventQueue>();
    fixture.network_api =
        std::make_unique<FakeQueueNetworkApi>(fixture.event_queue);
    auto source = std::make_shared<NodeStoreGraphSource>();
#ifdef COARSE_SEMANTIC_KEY_CACHE_TEST_REMOTE_MEM
    fixture.remote_api = std::make_unique<FakeRemoteMemoryApi>();
    fixture.sys = std::make_unique<Sys>(
        0, config_dir + "/workload", config_dir + "/comm_group.json",
        config_dir + "/" + system_file, fixture.remote_api.get(),
        fixture.network_api.get(),
        std::vector<int>{1}, std::vector<int>{1}, 1.0, 1.0, false,
        ExecutionMode::Online, source);
#else
    fixture.sys = std::make_unique<Sys>(
        0, config_dir + "/workload", config_dir + "/comm_group.json",
        config_dir + "/" + system_file, fixture.network_api.get(),
        std::vector<int>{1}, std::vector<int>{1}, 1.0, 1.0, false,
        ExecutionMode::Online, source);
#endif
    return fixture;
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);

    const std::string tmp_template = "/tmp/coarse_semantic_key_test_XXXXXX";
    std::vector<char> buffer(tmp_template.begin(), tmp_template.end());
    buffer.push_back('\0');
    const char* const made = mkdtemp(buffer.data());
    if (made == nullptr) {
        std::fprintf(stderr, "[coarse_semantic_key_cache_test] mkdtemp failed\n");
        return 1;
    }
    const std::string config_dir(made);

    std::filesystem::path repo(__FILE__);
    for (unsigned i = 0; i < 5; ++i) repo = repo.parent_path();
    std::ifstream coarse_hardware_in(
        repo / "extern/die_model/coarse/configs/example_hardware.json");
    if (!coarse_hardware_in.is_open()) {
        std::fprintf(stderr,
                     "[coarse_semantic_key_cache_test] cannot read the "
                     "coarse example hardware fixture\n");
        return 1;
    }
    nlohmann::json coarse_hardware;
    coarse_hardware_in >> coarse_hardware;
    write_text(config_dir + "/coarse_hardware.json",
               coarse_hardware.dump());
    // A second valid hardware: another row of the same core resource table
    // (double NoC width), so its resolved fingerprint must differ.
    nlohmann::json wide_hardware = coarse_hardware;
    wide_hardware["architecture"]["noc_width_bits"] = 512;
    write_text(config_dir + "/coarse_hardware_wide.json",
               wide_hardware.dump());
    write_text(config_dir + "/comm_group.json", "{}");
    write_text(config_dir + "/coarse_system.json", nlohmann::json{
        // joint alone fails fast without a scheduling policy; the other
        // four parse the key as optional, so one shared system file carries
        // a legal explicit value for all five.
        {"scheduling-policy", "FIFO"},
        {"compute-cost-model", "die"},
        {"die-model-profile", "coarse_die_v2"},
        {"die-model-config", config_dir + "/coarse_hardware.json"},
        {"local-mem-latency", 100}}.dump());

    Fixture fixture = make_fixture(config_dir, "coarse_system.json");
    if (fixture.sys->coarse_die_runtime == nullptr) {
        std::fprintf(stderr,
                     "[coarse_semantic_key_cache_test] no coarse runtime\n");
        return 1;
    }
    CoarseDieRuntime& runtime = *fixture.sys->coarse_die_runtime;

    const auto spec_a = legacy_ffn_spec(
        "batch_train_i1_1#s0", "batch_train_i1_1#s1",
        "transformer_pass_aggregated",
        "3321c0cd5543414ad55b4601497027130d0e0a32994a7e52cae3716e13424cda");
    // Same compile semantics, new train identity and bookkeeping strings.
    const auto spec_a_prime = legacy_ffn_spec(
        "batch_train_i9_7#s0", "batch_train_i9_7#s1", "other_emitter",
        "0000000000000000000000000000000000000000000000000000000000000000");
    auto spec_b = spec_a;
    spec_b["spans"][1]["kv_tokens"] = 5;

    const auto plan_a1 = runtime.compile(spec_a);
    const auto plan_a2 = runtime.compile(spec_a);
    const auto plan_a_prime = runtime.compile(spec_a_prime);
    const auto plan_b = runtime.compile(spec_b);

    expect_true(plan_a2.get() == plan_a1.get(),
                "A: identical task repeats hit the cached plan instance");
    expect_true(plan_a_prime.get() == plan_a1.get(),
                "B: identity-only change hits the cached plan instance");
    expect_true(plan_b.get() != plan_a1.get() &&
                    plan_b->task != plan_a1->task,
                "C: one semantic field change compiles a distinct plan");
    expect_true(plan_a1->task == plan_a_prime->task,
                "B: the reused plan carries the same normalized task");

    const nlohmann::json cache = runtime.plan_cache_counters();
    expect_true(cache.at("plan_cache_compile_hits") == 2 &&
                    cache.at("plan_cache_compile_misses") == 2 &&
                    cache.at("plan_cache_entries") == 2,
                "A/B/C: two semantic plans, two hits, two misses");
    expect_true(cache.at("plan_cache_key_ns_total").is_number_unsigned() &&
                    cache.at("plan_cache_key_ns_total")
                            .get<std::uint64_t>() > 0,
                "key generation time is observed as a positive uint64");

    // Different hardware fingerprint: the semantic task text is hardware
    // independent, so separation must come from the fingerprint the host
    // mixes into its own hash; a second runtime starts cold either way.
    const coarse_die::Hardware wide_hw =
        coarse_die::resolve_hardware(wide_hardware);
    expect_true(wide_hw.fingerprint != runtime.hardware().fingerprint,
                "D: the wide NoC fixture resolves to a different fingerprint");
    CoarseDieRuntime wide_runtime(fixture.sys.get(), wide_hw);
    const auto wide_plan = wide_runtime.compile(spec_a);
    const auto wide_hit = wide_runtime.compile(spec_a_prime);
    const nlohmann::json wide_cache = wide_runtime.plan_cache_counters();
    expect_true(wide_plan.get() != plan_a1.get() &&
                    wide_plan->hardware_fingerprint == wide_hw.fingerprint,
                "D: the other hardware compiles its own plan instance");
    expect_true(wide_hit.get() == wide_plan.get() &&
                    wide_cache.at("plan_cache_compile_hits") == 1 &&
                    wide_cache.at("plan_cache_compile_misses") == 1,
                "D: the second runtime hits inside its own cache only");

    // The host observation bundle merges the coarse Runtime's §4.2 call
    // counters; nothing was submitted, so they are present and zero.
    const nlohmann::json counters = runtime.counters();
    expect_true(counters.contains("next_event_calls") &&
                    counters.contains("advance_to_calls") &&
                    counters.contains("allocate_calls") &&
                    counters.contains("solver_pivots_total") &&
                    counters.contains("arbitration_paths") &&
                    counters.at("next_event_calls") == 0 &&
                    counters.at("allocate_calls") == 0,
                "E: runtime observation counters merge into counters()");
    expect_true(counters.contains("schedule_next_calls") &&
                    counters.at("schedule_next_calls") == 0,
                "E: schedule_next_calls is exposed and starts at zero");

    fixture.sys.reset();
    std::filesystem::remove_all(config_dir);
    if (!g_ok) {
        std::fprintf(stderr, "[coarse_semantic_key_cache_test] FAILED\n");
        return 1;
    }
    std::printf("[coarse_semantic_key_cache_test] all cases passed\n");
    return 0;
}
