/******************************************************************************
custom_only_system_construct_test.cc -- retired custom-collective keys
fail closed at parse time (P2-23 batch B2, 2026-09-28).

History: A2 (2026-09-28) fixed P2-8 by making a pure-custom system.json
(the shipped examples/system/custom_collectives/custom_collective.json
shape) constructible.  Batch B2 then retired the whole custom
collectives / feeder_v3 ET path (the offline static arm it rode on was
already removed on 2026-09-27), which supersedes that construct-repair:
the retired keys now fail closed AT PARSE TIME with an explicit
retirement message (CollectiveImplLookup::setup_collective_impl_from_
config), before any topology construction runs.

Contract pinned here:
  - a system.json containing "all-reduce-implementation-custom" (global
    custom key) exits cleanly (exit code 1, never a signal) with the
    "retired" critical message on stderr -- never constructs;
  - same for the per-node key "per-node-custom-implementation";
  - a native-only config ("all-reduce-implementation": ["ring"]) still
    constructs: the All_Reduce type gets a topology entry (dimension
    count preserved) and unconfigured types get no topology entry --
    looking one up now fails closed through Sys::sys_panic (exit 1 with a
    message naming the type and the absent system key), never a null
    return (F9 2026-09-29: the null flowed into generate_collective as a
    bare nullptr dereference).

Build: cmake target
       AstraSim_Analytical_Congestion_Aware_CustomOnlySystemConstructTest.
Run: build/astra_analytical/build_congestion_aware/bin/\
     AstraSim_Analytical_Congestion_Aware_CustomOnlySystemConstructTest
Exit code 0 on ALL PASS.
******************************************************************************/

#include "astra-sim/common/AstraNetworkAPI.hh"
#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using namespace AstraSim;
using AstraSim::ExecutionDriven::NodeStoreGraphSource;

namespace {

bool g_ok = true;
int g_checks = 0;

void expect(bool condition, const char* message) {
    ++g_checks;
    std::printf("%-78s -> %s\n", message, condition ? "PASS" : "FAIL");
    if (!condition) {
        g_ok = false;
    }
}

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

std::string write_temp_file(const std::string& pattern_prefix,
                            const std::string& contents) {
    std::string pattern = pattern_prefix;
    pattern += "XXXXXX";
    const int fd = ::mkstemp(pattern.data());
    if (fd < 0) {
        std::perror("mkstemp");
        std::exit(EXIT_FAILURE);
    }
    size_t offset = 0;
    while (offset < contents.size()) {
        const ssize_t written =
            ::write(fd, contents.data() + offset, contents.size() - offset);
        if (written > 0) {
            offset += static_cast<size_t>(written);
            continue;
        }
        if (written < 0 && errno == EINTR) {
            continue;
        }
        std::perror("write");
        ::close(fd);
        ::unlink(pattern.c_str());
        std::exit(EXIT_FAILURE);
    }
    ::close(fd);
    return pattern;
}

// One fork per retired custom key: the child must die through the
// fail-closed parse channel (clean exit 1 with the "retired" critical
// message on the logger sinks) -- never construct a Sys, never terminate
// by signal.  The child wires LoggerFactory to a temp dir first: without
// init_default_components the astraccl logger is a null sink and the
// retirement message would be dropped (production mains always init it).
void run_retired_key_child(const std::string& json, const char* exit_label,
                           const char* message_label) {
    const std::string system_config =
        write_temp_file("/tmp/astra-custom-retired-sys-", json);
    const std::string err_path =
        write_temp_file("/tmp/astra-custom-retired-err-", "");
    char dir_pattern[] = "/tmp/astra-custom-retired-log-XXXXXX";
    const char* made = ::mkdtemp(dir_pattern);
    if (made == nullptr) {
        std::perror("mkdtemp");
        std::exit(EXIT_FAILURE);
    }
    const std::string log_dir(made);
    const pid_t pid = ::fork();
    if (pid < 0) {
        std::perror("fork");
        std::exit(EXIT_FAILURE);
    }
    if (pid == 0) {
        // Child: route both console streams into the capture file, wire
        // the logger sinks, then attempt the construction that parse-time
        // fail-closed must stop.
        if (std::freopen(err_path.c_str(), "w", stdout) == nullptr ||
            std::freopen(err_path.c_str(), "w", stderr) == nullptr) {
            _exit(2);
        }
        AstraSim::LoggerFactory::init("empty", log_dir);
        auto* net = new FakeNetworkApi();
        auto source = std::make_shared<NodeStoreGraphSource>();
        Sys* sys = new Sys(0, "workload-unused", "empty", system_config, net,
                           std::vector<int>{1}, std::vector<int>{1}, 1.0, 1.0,
                           false,
                           AstraSim::ExecutionDriven::ExecutionMode::Online,
                           source);
        (void)sys;  // reaching here means the retired key was accepted
        std::exit(0);
    }
    int status = 0;
    if (::waitpid(pid, &status, 0) < 0) {
        std::perror("waitpid");
        std::exit(EXIT_FAILURE);
    }
    expect(WIFEXITED(status) && WEXITSTATUS(status) == 1, exit_label);

    std::ifstream err_in(log_dir + "/err.log");
    std::string err_text((std::istreambuf_iterator<char>(err_in)),
                         std::istreambuf_iterator<char>());
    if (err_text.find("retired") == std::string::npos) {
        std::ifstream capture_in(err_path);
        err_text += std::string((std::istreambuf_iterator<char>(capture_in)),
                                std::istreambuf_iterator<char>());
    }
    expect(err_text.find("retired") != std::string::npos, message_label);

    ::unlink(system_config.c_str());
    ::unlink(err_path.c_str());
    ::unlink((log_dir + "/err.log").c_str());
    ::unlink((log_dir + "/log.log").c_str());
    ::rmdir(log_dir.c_str());
}

// F9 (2026-09-29): a logical-topology lookup for an unconfigured collective
// type no longer returns nullptr (that null flowed into generate_collective
// as a bare segfault with no diagnosis). One fork per observation: the child
// re-builds the native-only Sys and touches the unconfigured type -- the
// process must exit 1 through Sys::sys_panic with the named diagnostic,
// never hand back a pointer the caller can dereference.
void run_unconfigured_type_child(const std::string& system_config) {
    const std::string err_path =
        write_temp_file("/tmp/astra-topology-missing-err-", "");
    char dir_pattern[] = "/tmp/astra-topology-missing-log-XXXXXX";
    const char* made = ::mkdtemp(dir_pattern);
    if (made == nullptr) {
        std::perror("mkdtemp");
        std::exit(EXIT_FAILURE);
    }
    const std::string log_dir(made);
    const pid_t pid = ::fork();
    if (pid < 0) {
        std::perror("fork");
        std::exit(EXIT_FAILURE);
    }
    if (pid == 0) {
        if (std::freopen(err_path.c_str(), "w", stdout) == nullptr ||
            std::freopen(err_path.c_str(), "w", stderr) == nullptr) {
            _exit(2);
        }
        AstraSim::LoggerFactory::init("empty", log_dir);
        auto* net = new FakeNetworkApi();
        auto source = std::make_shared<NodeStoreGraphSource>();
        Sys* sys = new Sys(0, "workload-unused", "empty", system_config, net,
                           std::vector<int>{1}, std::vector<int>{1}, 1.0, 1.0,
                           false,
                           AstraSim::ExecutionDriven::ExecutionMode::Online,
                           source);
        (void)sys;
        auto* topology = sys->get_logical_topology(ComType::Reduce_Scatter);
        // Reaching this line means the lookup handed back a pointer (null
        // or not) instead of failing closed -- fail the child loudly.
        std::printf("unconfigured-type lookup returned %p\n",
                    static_cast<void*>(topology));
        std::exit(0);
    }
    int status = 0;
    if (::waitpid(pid, &status, 0) < 0) {
        std::perror("waitpid");
        std::exit(EXIT_FAILURE);
    }
    expect(WIFEXITED(status) && WEXITSTATUS(status) == 1,
           "unconfigured collective lookup fails closed via exit(1), "
           "never a null return");

    std::ifstream err_in(log_dir + "/err.log");
    std::string err_text((std::istreambuf_iterator<char>(err_in)),
                         std::istreambuf_iterator<char>());
    if (err_text.find("no logical topology registered") == std::string::npos) {
        std::ifstream capture_in(err_path);
        err_text += std::string((std::istreambuf_iterator<char>(capture_in)),
                                std::istreambuf_iterator<char>());
    }
    expect(err_text.find("no logical topology registered for collective "
                         "ReduceScatter") != std::string::npos &&
               err_text.find("reduce-scatter-implementation") !=
                   std::string::npos,
           "unconfigured collective critical names the type and the "
           "absent system key");

    ::unlink(err_path.c_str());
    ::unlink((log_dir + "/err.log").c_str());
    ::unlink((log_dir + "/log.log").c_str());
    ::rmdir(log_dir.c_str());
}

// Native-only configs must keep constructing after the custom removal.
void run_native_construct_case() {
    const std::string system_config = write_temp_file(
        "/tmp/astra-native-sys-",
        "{\n"
        "  \"scheduling-policy\": \"FIFO\",\n"
        "  \"preferred-dataset-splits\": 1,\n"
        "  \"all-reduce-implementation\": [\"ring\"],\n"
        "  \"local-mem-bw\": 50\n"
        "}\n");
    // F9 child observation must run BEFORE this process builds its own Sys:
    // the construction below creates async spdlog loggers (get_logger ->
    // create_async -> a live thread-pool worker), and a forked child cannot
    // use that inherited pool -- the worker thread does not exist after
    // fork, so the child's sys_panic log path deadlocks on futexes. Forked
    // from the pre-Sys state, the child wires its own fresh pool first
    // (same ordering as run_retired_key_child, which is why those children
    // work).
    run_unconfigured_type_child(system_config);
    auto* net = new FakeNetworkApi();
    auto source = std::make_shared<NodeStoreGraphSource>();
    Sys* sys = nullptr;
    try {
        sys = new Sys(0, "workload-unused", "empty", system_config, net,
                      std::vector<int>{1}, std::vector<int>{1}, 1.0, 1.0,
                      false,
                      AstraSim::ExecutionDriven::ExecutionMode::Online,
                      source);
    } catch (const std::exception&) {
        sys = nullptr;  // sys_panic exits before this, defensive only
    }
    expect(sys != nullptr, "native-only system.json still constructs a Sys");
    expect(sys != nullptr &&
               sys->get_logical_topology(ComType::All_Reduce) != nullptr,
           "All_Reduce logical topology built from the native impl");
    expect(sys != nullptr &&
               sys->get_logical_topology(ComType::All_Reduce)
                       ->get_num_of_dimensions() == 1,
           "native topology carries one dimension");
    (void)::unlink(system_config.c_str());
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);

    run_retired_key_child(
        "{\n"
        "  \"all-reduce-implementation-custom\": "
        "[\"custom_allreduce_placeholder\"],\n"
        "  \"local-mem-bw\": 50\n"
        "}\n",
        "global custom key fails closed via exit(1), not construction",
        "global custom key critical names the retirement");

    run_retired_key_child(
        "{\n"
        "  \"per-node-custom-implementation\": "
        "\"per_node_placeholder.yml\",\n"
        "  \"local-mem-bw\": 50\n"
        "}\n",
        "per-node custom key fails closed via exit(1), not construction",
        "per-node custom key critical names the retirement");

    run_native_construct_case();

    std::printf("%d checks: %s\n", g_checks, g_ok ? "ALL PASS" : "FAILED");
    return g_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
