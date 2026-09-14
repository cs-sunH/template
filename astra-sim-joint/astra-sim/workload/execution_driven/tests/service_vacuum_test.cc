/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

service_vacuum_test.cc -- defect-A regression fixture (2026-08-16,
face主动测试错误分析.md 缺陷 A: 服务终判 finished() 的计数真空).

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

P0 turn-0 fix adaptation (2026-08-30): the calendar reader indexes the
whole file and queues EVERY turn-0 Submit during the first pump (the row
window no longer feeds rows gradually), so the historical "break mid-file
with completed=10/30" shape is no longer constructible. The defect CLASS
itself -- finished() consulted while queued-but-undrained Submit commands
exist -- is still faithfully reproduced: the legacy order below checks
finished() right after pump() queued the whole calendar and breaks with
completed=0/30 and 30 queued-but-undrained commands.

The fixture drives the REAL WindowedTraceReader / RequestIngress /
ServiceCoordinator / DecisionMailbox / EventQueue through both loop
orders on the same CSV (30 single-turn sessions, arrivals in 3 batches of
10 at ticks 1000/2000/3000):

  Part 1 (REPRO, legacy order): the loop breaks with queued-but-undrained
      commands > 0 and completed == 0 -- the counting vacuum, caught as
      an assertion (in production this was the silent early close). The
      audit_completion verdict must fail-closed (AccountMismatch).
  Part 2 (FIXED order): drain-first (drain whatever pump() just queued)
      plus the close-at-calendar-EOF boundary closes the vacuum; the loop
      ends with completed == total rows == 30 and verdict Ok.

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

// 30 single-turn sessions (P0 fix: one turn-0 row per session, blocks
// contiguous -- the structure validation fail-closes otherwise); arrivals
// in 3 batches of 10 (tick 1000/2000/3000). The calendar reader indexes
// the whole file and queues every turn-0
// Submit during the FIRST pump, so the vacuum window is "queued but not
// yet drained", not "read window boundary".
const char* kCsvHeader =
    "session_id,turn_index,request_id,prefill_length,decode_length,"
    "session_arrival_time_ns,inter_request_interval_ns";

std::string make_csv(const std::string& path) {
    std::ofstream out(path);
    out << kCsvHeader << "\n";
    for (int row = 0; row < 30; ++row) {
        const uint64_t arrival =
            1000 + static_cast<uint64_t>(row / 10) * 1000;  // 3 batches
        out << "s" << row << ",0,req_" << row << ",100,10," << arrival
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
    CompletionAuditVerdict verdict = CompletionAuditVerdict::Ok;
};

// One main-loop replica. `fixed_order` selects the production close-at-EOF
// ordering plus drain-after-pump. The legacy arm deliberately closes only the
// coordinator, leaving the old ingress producer gate open, so it can reproduce
// the historical queued-command vacuum without violating the new close gate.
// The arrival hook performs immediate no-node completion + window consumption.
RunResult run_loop(const std::string& csv, const bool fixed_order) {
    RunResult res;
    const auto eq = std::make_shared<EventQueue>();
    ServiceCoordinator svc;
    DecisionMailbox mailbox;
    RequestIngress ingress;
    ingress.bind(eq.get(), &mailbox, &svc);
    WindowedTraceReader reader(csv, ingress,
                               /*max_arrival_ns=*/0);
    ingress.set_arrival_hook([&](const RequestEnvelope& env) {
        // no-node fixture contract: the request completes at arrival
        reader.notify_consumed(env.queue_index);
        svc.on_request_completed();
    });
    if (!fixed_order) {
        // Historical behavior: the service could be closed while producers
        // still queued work. Do not use mark_input_closed() here: its new
        // linearized producer gate is precisely what this regression fixture
        // must bypass in order to model the old defect.
        svc.mark_input_closed();
    }
    reader.pump();  // initial pump before the loop (main_online.cc): the
                    // calendar reader queues ALL turn-0 Submits right here.
    if (fixed_order && reader.eof()) {
        ingress.mark_input_closed();
    }
    while (true) {
        if (!fixed_order) {
            // OLD (defect) order: finished() is consulted BEFORE the queued
            // Submits are drained. With the whole calendar freshly queued
            // this breaks immediately: completed=0 while 30 commands sit in
            // the ingress queue -- the counting vacuum.
            if (svc.finished()) {
                res.eof = reader.eof();
                res.queued_at_break = ingress.pending_command_count();
                res.broke_at_vacuum = res.queued_at_break > 0;
                break;
            }
        }
        ingress.drain_commands();
        reader.pump();
        if (fixed_order && !svc.input_closed() && reader.eof()) {
            // Production finite-input ordering: close only after every
            // calendar entry was submitted, so no later Submit is rejected.
            ingress.mark_input_closed();
        }
        if (fixed_order && ingress.pending_command_count() > 0) {
            // Defect-A fix: register what pump() just queued BEFORE the
            // finished() check (the vacuum drain).
            ingress.drain_commands();
        }
        if (svc.finished()) {
            res.eof = reader.eof();
            res.queued_at_break = ingress.pending_command_count();
            res.broke_at_vacuum = res.queued_at_break > 0;
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
    const CompletionAuditCounts counts{reader.total_data_rows(),
                                       reader.turn0_data_rows(),
                                       svc.accepted_request_count(),
                                       svc.completed_request_count(),
                                       reader.rejected_out_of_range()};
    res.verdict = audit_completion(counts);
    return res;
}

}  // namespace

int main() {
    const std::string root = "/tmp/service_vacuum_" +
                             std::to_string(::getpid());
    if (::mkdir(root.c_str(), 0755) != 0) {
        std::perror("mkdir");
        return 1;
    }
    const std::string csv = make_csv(root + "/queue.csv");

    // ---- Part 1: legacy order REPRODUCES the vacuum ----------------------
    const RunResult legacy = run_loop(csv, /*fixed_order=*/false);
    expect(legacy.broke_at_vacuum,
           "legacy order breaks with queued-but-undrained commands (the "
           "counting vacuum)");
    expect(legacy.eof,
           "legacy order: calendar reader already at EOF (whole file "
           "queued; the vacuum is undrained commands, not unread rows)");
    expect(legacy.queued_at_break == 30,
           "legacy order: all 30 queued-but-undrained commands exist at "
           "break");
    expect(legacy.completed == 0 && legacy.total_rows == 30,
           "legacy order completes 0/30 with the whole queue undelivered");
    expect(legacy.verdict != CompletionAuditVerdict::Ok,
           "legacy order audit verdict fail-closed (AccountMismatch: rows "
           "read but neither accepted nor completed)");
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
    expect(fixed.verdict == CompletionAuditVerdict::Ok,
           "fixed order audit verdict Ok");
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
