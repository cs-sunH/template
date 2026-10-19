/******************************************************************************
mem_dispatch_throw_test.cc -- MEM dispatch fail-closed negative regression
(remote-memory removal, 2026-09-24; plan §5 stage-2 negative case 4).

Constructs MEM_LOAD / MEM_STORE nodes directly in the NodeStore and hands
them to Workload::issue (no .et, no CSV, no full simulation): dispatch must
throw std::runtime_error ("MEM nodes have no issue path and must fail
loudly"), replacing the old NO_MEMORY_EXPANSION backend exit(1).  The
companion fail-closed is verified on the COMP side: a node carrying
remote_weight_bytes must throw in the roofline path, and an identical node
without it must dispatch cleanly (positive control -- proves a non-throwing
dispatch would be detected).

Build: cmake target AstraSim_Analytical_Congestion_Aware_MemDispatchThrowTest.
Run: build/astra_analytical/build_congestion_aware/bin/\
     AstraSim_Analytical_Congestion_Aware_MemDispatchThrowTest
Exit code 0 on ALL PASS.
****************************************************************************/

#include "astra-sim/common/AstraNetworkAPI.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace AstraSim;
using AstraSim::ExecutionDriven::NodeKind;
using AstraSim::ExecutionDriven::NodeStoreGraphSource;
using AstraSim::ExecutionDriven::NodeView;
using AstraSim::ExecutionDriven::OnlineNode;

namespace {

bool g_ok = true;

void expect(bool condition, const char* message) {
    std::printf("%-72s -> %s\n", message, condition ? "PASS" : "FAIL");
    if (!condition) {
        g_ok = false;
    }
}

// Fake network: construction-only; the negative cases throw before any
// event is scheduled, and the positive control leaves its single scheduled
// completion unpumped on purpose.
class FakeNetworkApi : public AstraNetworkAPI {
  public:
    FakeNetworkApi() : AstraNetworkAPI(0) {}

    int sim_send(void*, uint64_t, int, int, int, sim_request*,
                 void (*)(void*), void*) override {
        return 0;
    }

    // AstraNetworkAPI 纯虚面:桩无链路带宽语义,返回有限占位值即可。
    double get_BW_at_dimension(int /*dim*/) override { return 1.0; }

    int sim_recv(void*, uint64_t, int, int, int, sim_request*,
                 void (*)(void*), void*) override {
        return 0;
    }

    void sim_schedule(AstraSim::timespec_t, void (*)(void*), void*) override {}

    AstraSim::timespec_t sim_get_time() override {
        return {AstraSim::NS, 0};
    }
};

std::string write_minimal_system_config() {
    char path[] = "/tmp/astra_mem_dispatch_throw_XXXXXX";
    const int fd = ::mkstemp(path);
    if (fd < 0) {
        std::fprintf(stderr,
                     "[mem_dispatch_throw_test] mkstemp failed: %s\n",
                     std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
    // Roofline on with valid divisors: the COMP remote-weight case must
    // reach the shrunk roofline throw (not the missing-config exit). The
    // four *-implementation keys are mandatory: Sys fails closed
    // (GeneralComplexTopology, empty implementation table) without them,
    // and a single "ring" matches this fixture's one-dimension {1} shape.
    const std::string contents =
        "{\"scheduling-policy\":\"FIFO\",\"roofline-enabled\":1,"
        "\"peak-perf\":1000,\"local-mem-bw\":3000,"
        "\"local-mem-latency\":100,\"hbm-bandwidth-contention\":false,"
        "\"all-reduce-implementation\":[\"ring\"],"
        "\"all-gather-implementation\":[\"ring\"],"
        "\"reduce-scatter-implementation\":[\"ring\"],"
        "\"all-to-all-implementation\":[\"ring\"]}";
    if (::write(fd, contents.c_str(), contents.size()) < 0) {
        std::fprintf(stderr,
                     "[mem_dispatch_throw_test] config write failed: %s\n",
                     std::strerror(errno));
        ::close(fd);
        ::unlink(path);
        std::exit(EXIT_FAILURE);
    }
    ::close(fd);
    return path;
}

// One fresh Sys per case: a thrown dispatch never completes its node, so
// the in-flight hardware counters must not be shared across cases.  Sys is
// leaked deliberately (same pattern as local_hbm_bandwidth_model_test.cc):
// Sys::all_sys is process-global static state, and nothing is pumped, so
// teardown has nothing useful to do.
Workload* make_case_workload(const std::string& system_config,
                             std::shared_ptr<NodeStoreGraphSource>& source) {
    auto* net = new FakeNetworkApi();
    source = std::make_shared<NodeStoreGraphSource>();
    auto* sys = new Sys(0, "workload-unused", "empty", system_config, net,
                        std::vector<int>{1}, std::vector<int>{1}, 1.0, 1.0,
                        false, AstraSim::ExecutionDriven::ExecutionMode::Online,
                        source);
    return sys->workload;
}

uint64_t add_node(NodeStoreGraphSource& source, OnlineNode node) {
    node.rank = 0;
    const uint64_t id = source.store().add_node(std::move(node));
    return id;
}

// Returns the caught message; sets *threw. A clean return is reported as
// "no throw" so the caller can fail the case explicitly.
std::string issue_and_catch(Workload& workload,
                            NodeStoreGraphSource& source,
                            uint64_t node_id,
                            bool* threw) {
    *threw = false;
    const auto* view = source.lookup_ptr(node_id);
    if (view == nullptr) {
        return "<node lookup failed>";
    }
    try {
        workload.issue(*view);
    } catch (const std::runtime_error& e) {
        *threw = true;
        return e.what();
    }
    return "<no throw>";
}

void run_mem_case(const std::string& system_config,
                  NodeKind kind,
                  uint64_t node_type,
                  const char* label) {
    std::shared_ptr<NodeStoreGraphSource> source;
    Workload* workload = make_case_workload(system_config, source);

    OnlineNode node;
    node.kind = kind;
    node.node_type = node_type;  // raw ChakraProtoMsg::NodeType value
    node.name = label;
    node.compute.tensor_size = 1024;
    const uint64_t id = add_node(*source, std::move(node));

    bool threw = false;
    const std::string message =
        issue_and_catch(*workload, *source, id, &threw);
    expect(threw, label);
    const bool loud =
        message.find("MEM_LOAD/MEM_STORE") != std::string::npos &&
        message.find("remote memory") != std::string::npos;
    expect(loud, "MEM dispatch throw names the removed remote memory backend");
}

// The production issue paths (commit-time issue_rank callback and the
// post-commit deferred issue pass) call issue_dep_free_nodes(), not
// Workload::issue directly.  A node that throws in dispatch must propagate
// through the pass WITH its identity attached (id/type/name): the online
// fatal channel (main_online.cc) can then name the offending node, the
// same information the commit-period try/catch surfaces.  The original
// throw message must be preserved verbatim inside the enriched one.
void run_mem_dep_free_pass_case(const std::string& system_config,
                                NodeKind kind,
                                uint64_t node_type,
                                const char* label) {
    std::shared_ptr<NodeStoreGraphSource> source;
    Workload* workload = make_case_workload(system_config, source);

    OnlineNode node;
    node.kind = kind;
    node.node_type = node_type;  // raw ChakraProtoMsg::NodeType value
    node.name = label;
    node.compute.tensor_size = 1024;
    const uint64_t id = add_node(*source, std::move(node));

    std::string message = "<no throw>";
    bool threw = false;
    try {
        workload->issue_dep_free_nodes();
    } catch (const std::runtime_error& e) {
        threw = true;
        message = e.what();
    }
    expect(threw, label);
    const bool loud =
        message.find("MEM_LOAD/MEM_STORE") != std::string::npos &&
        message.find("remote memory") != std::string::npos;
    expect(loud, "issue pass preserves the original MEM throw message");
    const bool named =
        message.find("node issue failed") != std::string::npos &&
        message.find("id=" + std::to_string(id)) != std::string::npos &&
        message.find("type=" + std::to_string(node_type)) !=
            std::string::npos;
    expect(named, "issue pass names the failing node (id/type)");
}

void run_comp_remote_weight_case(const std::string& system_config,
                                 bool with_remote_weight) {
    std::shared_ptr<NodeStoreGraphSource> source;
    Workload* workload = make_case_workload(system_config, source);

    OnlineNode node;
    node.kind = NodeKind::Compute;
    node.node_type = 4;  // ChakraProtoMsg::COMP_NODE
    node.name = with_remote_weight ? "comp_remote_weight" : "comp_plain";
    node.compute.num_ops = 1000;
    node.compute.tensor_size = 512;
    node.compute.has_remote_weight_bytes = with_remote_weight;
    node.compute.remote_weight_bytes = 4096;
    const uint64_t id = add_node(*source, std::move(node));

    bool threw = false;
    const std::string message =
        issue_and_catch(*workload, *source, id, &threw);
    if (with_remote_weight) {
        expect(threw, "COMP with remote_weight_bytes throws in roofline path");
        const bool loud =
            message.find("remote operand pipeline loads") != std::string::npos;
        expect(loud, "roofline throw names the unsupported remote pipeline");
    } else {
        expect(!threw, "plain COMP (no remote_weight_bytes) dispatches");
    }
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string system_config = write_minimal_system_config();

    run_mem_case(system_config, NodeKind::MemLoad, 2, "MEM_LOAD dispatch");
    run_mem_case(system_config, NodeKind::MemStore, 3, "MEM_STORE dispatch");
    run_mem_dep_free_pass_case(system_config, NodeKind::MemLoad, 2,
                               "MEM_LOAD issue pass");
    run_mem_dep_free_pass_case(system_config, NodeKind::MemStore, 3,
                               "MEM_STORE issue pass");
    run_comp_remote_weight_case(system_config, true);
    run_comp_remote_weight_case(system_config, false);

    if (::unlink(system_config.c_str()) != 0) {
        std::fprintf(stderr,
                     "[mem_dispatch_throw_test] unlink failed: %s\n",
                     std::strerror(errno));
        return EXIT_FAILURE;
    }
    if (g_ok) {
        std::printf("ALL PASS\n");
        return EXIT_SUCCESS;
    }
    std::printf("FAILED\n");
    return EXIT_FAILURE;
}
