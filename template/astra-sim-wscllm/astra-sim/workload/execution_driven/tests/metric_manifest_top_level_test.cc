/******************************************************************************
metric_manifest_top_level_test.cc -- manifest top-level type guard
regression (P2, 2026-09-28).

load_manifest guards the types of every optional top-level key (and every
sub-key of requests/planner_memory_peaks/node_events_by_rank), but a
manifest whose TOP LEVEL is not a JSON object (e.g. a bare array) reached
json::value<>() unguarded: the resulting nlohmann type_error escapes
load_manifest, nothing catches it, and std::terminate kills the process
with SIGABRT -- bypassing this file's own fail-closed channel
(fatal_metrics_error: [METRIC][ERROR] on stderr + exit(1)).

The guard must route the failure through that channel instead.  Verified
in a forked child: the child feeds a bare-array manifest to
MetricCollector::initialize and the parent requires a CLEAN exit(1)
(WIFEXITED, status 1) -- never a signal termination.

Build: cmake target AstraSim_Analytical_Congestion_Aware_MetricManifestTopLevelTest.
Run: build/astra_analytical/build_congestion_aware/bin/\
     AstraSim_Analytical_Congestion_Aware_MetricManifestTopLevelTest
Exit code 0 on ALL PASS.
******************************************************************************/

#include "astra-sim/workload/MetricCollector.hh"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/wait.h>
#include <string>
#include <unistd.h>

#include <json/json.hpp>

namespace {

bool g_ok = true;
int g_checks = 0;

void expect(bool condition, const char* message) {
    ++g_checks;
    if (!condition) {
        std::fprintf(stderr,
                     "[metric_manifest_top_level_test] FAIL: %s\n", message);
        g_ok = false;
    }
}

std::string write_manifest(const std::string& contents) {
    char pattern[] = "/tmp/astra-metric-toplevel-XXXXXX";
    const int fd = ::mkstemp(pattern);
    if (fd < 0) {
        std::fprintf(stderr, "mkstemp failed: %s\n", std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
    size_t offset = 0;
    while (offset < contents.size()) {
        const ssize_t written = ::write(
            fd, contents.data() + offset, contents.size() - offset);
        if (written > 0) {
            offset += static_cast<size_t>(written);
            continue;
        }
        if (written < 0 && errno == EINTR) {
            continue;
        }
        std::fprintf(stderr, "write manifest failed: %s\n",
                     std::strerror(errno));
        ::close(fd);
        ::unlink(pattern);
        std::exit(EXIT_FAILURE);
    }
    if (::close(fd) != 0) {
        std::fprintf(stderr, "close manifest failed: %s\n",
                     std::strerror(errno));
        ::unlink(pattern);
        std::exit(EXIT_FAILURE);
    }
    return pattern;
}

// One fork per rejected input; the child either dies by signal (the
// uncaught type_error the guard removes) or exits through the fail-closed
// channel with status 1.
void run_rejected_child(const std::string& contents, const char* label) {
    const std::string manifest_path = write_manifest(contents);
    const pid_t pid = ::fork();
    if (pid < 0) {
        std::fprintf(stderr, "fork failed: %s\n", std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
    if (pid == 0) {
        // Child: initialize must fail closed (exit 1), never abort.
        AstraSim::MetricCollector::instance().initialize(manifest_path,
                                                         "full");
        // Reaching this line means the malformed manifest was accepted.
        std::exit(0);
    }
    int status = 0;
    if (::waitpid(pid, &status, 0) < 0) {
        std::fprintf(stderr, "waitpid failed: %s\n", std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
    expect(WIFEXITED(status) && WEXITSTATUS(status) == 1, label);
    ::unlink(manifest_path.c_str());
}

}  // namespace

int main() {
    run_rejected_child("[1, 2, 3]\n",
                       "bare-array manifest fails closed via exit(1), "
                       "not std::terminate");
    run_rejected_child("\"just a string\"\n",
                       "string manifest fails closed via exit(1), "
                       "not std::terminate");
    std::fprintf(stderr,
                 "[metric_manifest_top_level_test] %d checks: %s\n",
                 g_checks, g_ok ? "PASS" : "FAIL");
    return g_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
