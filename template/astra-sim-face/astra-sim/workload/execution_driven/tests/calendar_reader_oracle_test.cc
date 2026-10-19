/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

calendar_reader_oracle_test.cc -- P0 turn-0 fix V2 equivalence oracle
(2026-08-30; 2问题分析与解决方案kimi.md §4.3/§6 P0 reader 验收).

Proves: for the SAME request queue CSV, the NEW calendar reader and the OLD
(verbatim baseline copy) row-window reader in its UNBOUNDED arm
(the integrity control arm) produce an
ELEMENTWISE IDENTICAL turn-0 arrival fire sequence -- the ordered list of
(alarm_tick, queue_index) pairs as the real RequestIngress drain +
EventQueue actually fire them. This is the load-bearing equivalence
argument (WindowedTraceReader.hh item 4): EventQueue fires a shared tick's
EventList in insertion order; the old unbounded arm inserts turn-0 alarms
in ROW order, the calendar arm in (arrival, queue_index) order, and within
an equal-alarm group those orders coincide (queue_index ascending), so the
sequences must match on ANY well-formed input -- including inputs whose
turn-0 arrivals are non-monotonic in file order (the full TraceLab queue),
where the old BOUNDED arm was defective (491 clamped Submits on the real
run) while the old UNBOUNDED arm was merely wasteful.

Additionally asserts on the NEW arm only: late_static_submit == 0 and every
turn-0 reader_discovered_tick == 0 (the fix's own gates).

Usage:
  AstraSim_Analytical_Congestion_Aware_CalendarOracleTest [queue_csv]

Without arguments a synthetic out-of-order fixture is used (CI-hermetic).
With a real queue path (e.g. the full 22,816-row TraceLab queue) the same
assertions run on the real input; the provenance stats are printed so the
run record can quote them (turn0 count, file-order adjacent inversions,
arrival min/max).

Ingress capacity (review 2026-09-26 cpp-ed-tests-a-4): each arm constructs
its RequestIngress with an oversized command queue so one unbounded pump
never hits the 4096-command production default (which would fail the
legacy arm's enqueue or park the calendar arm's cursor mid-calendar and
truncate the comparison). The real 22,816-row TraceLab queue carries 496
turn-0 rows and fits the production default, but the oracle must not
depend on that ratio.

Build: the CMake target
AstraSim_Analytical_Congestion_Aware_CalendarOracleTest.
*******************************************************************************/

#include <cstdio>
#include <fstream>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include "astra-sim/workload/execution_driven/DecisionMailbox.hh"
#include "astra-sim/workload/execution_driven/RequestIngress.hh"
#include "astra-sim/workload/execution_driven/ServiceCoordinator.hh"
#include "astra-sim/workload/execution_driven/WindowedTraceReader.hh"
#include "astra-sim/workload/execution_driven/tests/LegacyOracleWindowedTraceReader.hh"
#include "astra-sim/workload/execution_driven/tests/fixture_provenance_sidecar.hh"
#include "common/EventQueue.h"

using namespace AstraSim::ExecutionDriven;
using namespace NetworkAnalytical;

namespace {

bool g_ok = true;

void expect(const bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "[calendar_oracle_test] FAIL: %s\n", what);
        g_ok = false;
    }
}

struct FireRecord {
    EventTime tick;
    int64_t queue_index;
};

// Drives one full reader arm on its OWN ingress/event queue and captures
// the (tick, queue_index) sequence in real fire order through the arrival
// hook (the hook runs inside RequestIngress::arrival_cb at the alarm tick).
template <typename Reader>
std::vector<FireRecord> run_arm(RequestIngress& ingress,
                                EventQueue& eq, Reader& reader) {
    std::vector<FireRecord> seq;
    ingress.set_arrival_hook([&](const RequestEnvelope& env) {
        seq.push_back(FireRecord{eq.get_current_time(), env.queue_index});
    });
    reader.pump();
    ingress.drain_commands();
    while (!eq.finished()) {
        eq.proceed();
    }
    return seq;
}

std::vector<FireRecord> run_legacy_arm(const std::string& csv) {
    EventQueue eq;
    DecisionMailbox mailbox;
    ServiceCoordinator svc;
    // Oversized command queue: see the ingress-capacity note in the header.
    RequestIngress ingress(/*capacity=*/1u << 20);
    ingress.bind(&eq, &mailbox, &svc);
    // UNBOUNDED arm: explicit high_water=0 reads the whole file in one pump
    // and queues every turn-0 Submit in ROW order (the pre-P0 full-pass
    // control arm; the constructor default is 128 = bounded, which would
    // clamp the real-queue argv[1] comparison into a false failure).
    LegacyOracleWindowedTraceReader reader(csv, ingress, 0);
    return run_arm(ingress, eq, reader);
}

void compare_arms(const std::string& csv, const char* label) {
    const std::vector<FireRecord> legacy = run_legacy_arm(csv);

    EventQueue eq;
    DecisionMailbox mailbox;
    ServiceCoordinator svc;
    // Oversized command queue: see the ingress-capacity note in the header
    // (a full production-default queue parks the calendar cursor and
    // truncates this arm without a second pump).
    RequestIngress ingress(/*capacity=*/1u << 20);
    ingress.bind(&eq, &mailbox, &svc);
    WindowedTraceReader reader(csv, ingress);
    const std::vector<FireRecord> calendar = run_arm(ingress, eq, reader);

    std::printf("[oracle:%s] turn-0 fire sequence length: legacy=%zu "
                "calendar=%zu\n",
                label, legacy.size(), calendar.size());
    expect(legacy.size() == calendar.size() && !legacy.empty(),
           "both arms fired the same nonzero number of turn-0 arrivals");
    bool elementwise = legacy.size() == calendar.size();
    for (size_t i = 0; elementwise && i < legacy.size(); ++i) {
        if (legacy[i].tick != calendar[i].tick ||
            legacy[i].queue_index != calendar[i].queue_index) {
            elementwise = false;
            std::fprintf(stderr,
                         "[oracle:%s] first divergence at #%zu: legacy "
                         "(tick=%llu q=%lld) calendar (tick=%llu q=%lld)\n",
                         label, i,
                         (unsigned long long)legacy[i].tick,
                         (long long)legacy[i].queue_index,
                         (unsigned long long)calendar[i].tick,
                         (long long)calendar[i].queue_index);
        }
    }
    expect(elementwise,
           "legacy unbounded vs calendar fire sequences are elementwise "
           "identical (incl. same-tick group order)");
    // New-arm-only gates (the fix's own invariants).
    expect(ingress.late_static_submit_count() == 0,
           "calendar arm: late_static_submit == 0");
    const auto summary = reader.audit_static_arrivals();
    expect(summary.gate_ok, "calendar arm: arrival gate ok");
    expect(summary.nonzero_delay_rows == 0,
           "calendar arm: every turn-0 ingress_delay == 0 (t0 boundary "
           "exempt)");
    bool discovered_zero = true;
    for (const auto& row : reader.arrival_audit()) {
        if (row.reader_discovered_tick != 0) {
            discovered_zero = false;
        }
    }
    expect(discovered_zero, "calendar arm: every turn-0 discovered at tick 0");
    std::printf("[oracle:%s] rows: total=%llu turn0=%llu\n",
                label, (unsigned long long)reader.total_data_rows(),
                (unsigned long long)reader.turn0_data_rows());
    std::printf("[oracle:%s] PASS: sequences identical (%zu arrivals), "
                "late_static_submit=0, gate ok\n",
                label, calendar.size());
}

// Synthetic hermetic fixture: out-of-order turn-0 across session blocks,
// a same-tick group, an arrival==0 row, multi-turn blocks.
void write_synthetic(const std::string& csv) {
    // P1 fix (2026-09-28, ported to this repo 2026-10-07): the provenance
    // gate is fail-closed on an absent sidecar, so the synthetic fixture
    // ships its matching record too (written by the shared helper; raw
    // rows identical to before).
    FixtureProvenance::write_queue_fixture(
        csv,
        {
            // File-order turn-0 arrivals: 500, 100, 300, 300, 0, 700
            // (blocks contiguous, arrivals wildly non-monotonic).
            "s0,0,s0_r0,100,50,500,,d",
            "s0,1,s0_r1,100,50,,90,d",
            "s1,0,s1_r0,100,50,100,,d",
            "s2,0,s2_r0,100,50,300,,d",
            "s2,1,s2_r1,100,50,,50,d",
            "s3,0,s3_r0,100,50,300,,d",
            "s4,0,s4_r0,100,50,0,,d",
            "s5,0,s5_r0,100,50,700,,d",
        });
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc > 1) {
        // Real queue (e.g. the full 22,816-row TraceLab queue).
        compare_arms(argv[1], argv[1]);
    } else {
        const std::string csv =
            "/tmp/calendar_oracle_fixture_" + std::to_string(::getpid()) +
            ".csv";
        write_synthetic(csv);
        compare_arms(csv, "synthetic");
        std::remove(csv.c_str());
        std::remove((csv + ".provenance.json").c_str());
    }
    if (!g_ok) {
        std::fprintf(stderr,
                     "[calendar_oracle_test] FAIL: see messages above\n");
        return 1;
    }
    std::printf("[calendar_oracle_test] ALL PASS\n");
    return 0;
}
