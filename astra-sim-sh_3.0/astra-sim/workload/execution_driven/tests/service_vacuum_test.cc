/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

service_vacuum_test.cc -- defect-A regression fixture (2026-08-16,
synced from face-defectfix2-done; source analysis: face主动测试错误分析.md
缺陷 A: 服务终判 finished() 的计数真空).

Reproduces, deterministically and unit-level, the F2/F3/F4 race window:
the online main loop's OLD order was

    drain_commands(); windowed.pump(); if (svc.finished()) break;

pump() only QUEUES Submit commands into the bounded ingress queue; they
register as pending alarms (on_alarm_scheduled) on the NEXT drain. On the
loop turn where (a) every previously drained alarm has fired, (b) the
arrival hook completed the requests immediately (no-node fixture
contract), and (c) pump() just topped the window up with NEW rows -- the
finished() check saw active==0 && pending_alarm==0 && input closed ==
true while queued-but-undrained commands existed, and ended the run with
the CSV tail undelivered (the field signature: ack_count != delivery_count,
恒差 1, and completed << total rows).

The fixture drives the REAL WindowedTraceReader / RequestIngress /
ServiceCoordinator / DecisionMailbox / EventQueue through both loop
orders on the same CSV (30 turn-0 rows, arrivals in 3 batches of 10,
high_water=10 so the file is read in exactly 3 pumps):

  Part 1 (REPRO, legacy order):  the loop breaks with reader NOT at EOF,
      completed=10/30, queued-but-undrained commands > 0 -- the vacuum,
      caught as an assertion (in production this was the silent early
      close). The audit_completion verdict must be Incomplete.
  Part 2 (FIXED order): the extra "drain whatever pump() just queued"
      (pending_command_count() > 0 -> drain_commands()) closes the
      vacuum; the loop ends only at EOF with completed == total rows ==
      30 and verdict Ok.

Build: CMake target AstraSim_Analytical_Congestion_Aware_ServiceVacuumTest.
Exit code 0 = both parts behaved as described (legacy reproduces, fixed
is green).
*******************************************************************************/

#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "astra-sim/workload/execution_driven/DecisionMailbox.hh"
#include "astra-sim/workload/execution_driven/RequestIngress.hh"
#include "astra-sim/workload/execution_driven/ServiceCoordinator.hh"
#include "astra-sim/workload/execution_driven/WindowedTraceReader.hh"
#include <astra-network-analytical/common/EventQueue.h>

using namespace AstraSim::ExecutionDriven;
using namespace NetworkAnalytical;

namespace {

bool g_ok = true;

void expect(bool cond, const std::string& what) {
    if (!cond) {
        std::fprintf(stderr, "[service_vacuum_test] FAIL: %s\n", what.c_str());
        g_ok = false;
    }
}

// 30 turn-0 data rows; arrivals in 3 batches of 10 (tick 1000/2000/3000).
// high_water=10 => the reader consumes the file in exactly 3 pumps, and
// between pumps the "all alarms fired + all requests completed" state is
// reached while the next 10 rows are freshly queued -- the vacuum window.
const char* kCsvHeader =
    "session_id,turn_index,request_id,prefill_length,decode_length,"
    "session_arrival_time_ns,inter_request_interval_ns";

std::string make_csv(const std::string& path) {
    std::ofstream out(path);
    out << kCsvHeader << "\n";
    for (int row = 0; row < 30; ++row) {
        const uint64_t arrival =
            1000 + static_cast<uint64_t>(row / 10) * 1000;  // 3 batches
        out << "s" << (row / 10) << ",0,req_" << row << ",100,10," << arrival
            << ",0\n";
    }
    return path;
}

struct RunResult {
    bool broke_at_vacuum = false;   // legacy: break while pump queued work
    uint64_t completed = 0;
    uint64_t total_rows = 0;
    bool eof = false;
    size_t queued_at_break = 0;
    // sh_3.0 adaptation: this repo's run-end audit lives in main_online
    // (completed_request_count != data_rows => Error + EXIT_FAILURE), not
    // as a WindowedTraceReader audit_completion API (that is the
    // wscllm/sh_1.0 fork's shape). The fixture reproduces the repo's own
    // verdict: completed == rows read at break.
    bool audit_ok = false;
};

// One main-loop replica. `fixed_order` selects drain-after-pump (the fix).
// Everything else mirrors main_online.cc: bind, close-at-startup, initial
// pump, arrival hook = immediate no-node completion + window consumption.
RunResult run_loop(const std::string& csv, const bool fixed_order) {
    RunResult res;
    const auto eq = std::make_shared<EventQueue>();
    ServiceCoordinator svc;
    DecisionMailbox mailbox;
    RequestIngress ingress;
    ingress.bind(eq.get(), &mailbox, &svc);
    WindowedTraceReader reader(csv, ingress, /*high_water=*/10,
                               /*max_arrival_ns=*/0);
    ingress.set_arrival_hook([&](const RequestEnvelope& env) {
        // no-node fixture contract: the request completes at arrival
        reader.notify_consumed(env.queue_index);
        svc.on_request_completed();
    });
    ingress.mark_input_closed();  // official runners: --close-input

    reader.pump();  // initial top-up before the loop (main_online.cc)
    while (true) {
        ingress.drain_commands();
        reader.pump();
        if (fixed_order && ingress.pending_command_count() > 0) {
            // Defect-A fix: register what pump() just queued BEFORE the
            // finished() check (the vacuum drain).
            ingress.drain_commands();
        }
        if (svc.finished()) {
            res.eof = reader.eof();
            res.queued_at_break = ingress.pending_command_count();
            res.broke_at_vacuum = !reader.eof();
            break;
        }
        if (eq->finished()) {
            // Unit fixture: with the input closed and no tick-end gate, an
            // empty queue here is either the (legacy) vacuum break above or
            // an internal inconsistency -- the production fail-closed path
            // is exercised by the wakeup-guard end-to-end fixture.
            break;
        }
        eq->proceed();
        if (mailbox.has_decision_work()) {
            (void)mailbox.drain();  // delivery stand-in: consume the epoch
        }
    }
    res.completed = svc.completed_request_count();
    res.total_rows = reader.data_rows();
    // sh_3.0 repo audit semantics (main_online run-end gate):
    // completed == expected == data_rows, else the run FAILs.
    res.audit_ok = (res.completed == res.total_rows);
    return res;
}

}  // namespace

int main() {
    const std::string root = "/tmp/sh30_service_vacuum_" +
                             std::to_string(::getpid());
    if (::mkdir(root.c_str(), 0755) != 0) {
        std::perror("mkdir");
        return 1;
    }
    const std::string csv = make_csv(root + "/queue.csv");

    // ---- Part 1: legacy order REPRODUCES the vacuum ----------------------
    const RunResult legacy = run_loop(csv, /*fixed_order=*/false);
    expect(legacy.broke_at_vacuum,
           "legacy order breaks while the reader is not at EOF (vacuum)");
    expect(!legacy.eof, "legacy order: reader NOT at EOF at break");
    expect(legacy.queued_at_break > 0,
           "legacy order: queued-but-undrained commands exist at break");
    expect(legacy.completed == 10 && legacy.total_rows == 20,
           "legacy order completes 10 with only 20/30 rows read (the "
           "undelivered tail)");
    expect(!legacy.audit_ok,
           "legacy order audit FAIL (completed != data rows read: the "
           "undelivered tail -- this repo's main_online run-end gate)");
    std::printf("[service_vacuum_test] legacy order: VACUUM reproduced "
                "(break at completed=%llu/%llu, queued=%zu, eof=%d)\n",
                static_cast<unsigned long long>(legacy.completed),
                static_cast<unsigned long long>(legacy.total_rows),
                legacy.queued_at_break, legacy.eof ? 1 : 0);

    // ---- Part 2: fixed order closes the vacuum ---------------------------
    const RunResult fixed = run_loop(csv, /*fixed_order=*/true);
    expect(!fixed.broke_at_vacuum, "fixed order does not break pre-EOF");
    expect(fixed.eof, "fixed order: reader reached EOF");
    expect(fixed.queued_at_break == 0,
           "fixed order: nothing queued-but-undrained at break");
    expect(fixed.completed == 30 && fixed.total_rows == 30,
           "fixed order completes 30/30 (末笔交付前全部登记)");
    expect(fixed.audit_ok,
           "fixed order audit Ok (completed == data rows)");
    std::printf("[service_vacuum_test] fixed order: green "
                "(completed=%llu/%llu, queued=%zu, eof=%d)\n",
                static_cast<unsigned long long>(fixed.completed),
                static_cast<unsigned long long>(fixed.total_rows),
                fixed.queued_at_break, fixed.eof ? 1 : 0);

    const std::string rm = "rm -rf '" + root + "'";
    const int rc = std::system(rm.c_str());
    (void)rc;
    if (!g_ok) {
        std::fprintf(stderr, "[service_vacuum_test] FAIL (see above)\n");
        return 1;
    }
    std::printf("[service_vacuum_test] ALL PASS: vacuum reproduced on the "
                "legacy order, closed by the drain-after-pump fix\n");
    return 0;
}
