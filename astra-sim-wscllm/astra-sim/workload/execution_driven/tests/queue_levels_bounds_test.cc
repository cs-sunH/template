/******************************************************************************
queue_levels_bounds_test.cc -- QueueLevels out-of-range guard regression
(P2, 2026-09-28).

The three QueueLevels accessors indexed levels[level] with no bounds
check.  A communicator whose dimension count exceeds the system's queue
dimensions reaches get_next_queue_at_level(dim) with an out-of-range
level and reads the vector out of bounds (UB; upstream usually guards,
so this is the hardening backstop).

The guard fails closed through the same channel as the other system-layer
bounds checks (GeneralComplexTopology::get_num_of_nodes_in_dimension):
critical log line + exit(1).  Verified in a forked child: the parent
requires a CLEAN exit(1) (WIFEXITED, status 1) -- never a signal
termination from the raw out-of-bounds read.  In-range access must keep
returning the exact handler windows unchanged.

Build: cmake target AstraSim_Analytical_Congestion_Aware_QueueLevelsBoundsTest.
Run: build/astra_analytical/build_congestion_aware/bin/\
     AstraSim_Analytical_Congestion_Aware_QueueLevelsBoundsTest
Exit code 0 on ALL PASS.
******************************************************************************/

#include "astra-sim/system/QueueLevels.hh"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>

using AstraSim::AstraNetworkAPI;
using AstraSim::QueueLevels;

namespace {

bool g_ok = true;
int g_checks = 0;

void expect(bool condition, const char* message) {
    ++g_checks;
    if (!condition) {
        std::fprintf(stderr, "[queue_levels_bounds_test] FAIL: %s\n",
                     message);
        g_ok = false;
    }
}

// One fork per out-of-range access; the child must die through the
// fail-closed channel (exit 1), never by signal from the raw read.
void run_out_of_range_child(int level, const char* label) {
    const pid_t pid = ::fork();
    if (pid < 0) {
        std::fprintf(stderr, "fork failed: %s\n", std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
    if (pid == 0) {
        QueueLevels levels(std::vector<int>{2, 2}, 0);
        (void)levels.get_next_queue_at_level(level);
        // Reaching this line means the out-of-range level was accepted.
        std::exit(0);
    }
    int status = 0;
    if (::waitpid(pid, &status, 0) < 0) {
        std::fprintf(stderr, "waitpid failed: %s\n", std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
    expect(WIFEXITED(status) && WEXITSTATUS(status) == 1, label);
}

}  // namespace

int main() {
    // In-range contract unchanged: the level windows and the rotating
    // queue ids are exactly the constructor's [start, end] ranges.
    QueueLevels levels(std::vector<int>{2, 3}, 0);
    expect(levels.levels.size() == 2, "two levels constructed");
    const auto l0_a = levels.get_next_queue_at_level(0);
    const auto l0_b = levels.get_next_queue_at_level(0);
    const auto l0_c = levels.get_next_queue_at_level(0);
    const auto l1_a = levels.get_next_queue_at_level(1);
    expect(l0_a.first == 0 && l0_b.first == 1 && l0_c.first == 0,
           "level 0 window is [0,1] and rotates");
    expect(l1_a.first == 2, "level 1 window starts at 2");

    run_out_of_range_child(5, "level 5 over 2 levels exits 1, no signal");
    run_out_of_range_child(-1, "negative level exits 1, no signal");

    std::fprintf(stderr, "[queue_levels_bounds_test] %d checks: %s\n",
                 g_checks, g_ok ? "PASS" : "FAIL");
    return g_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
