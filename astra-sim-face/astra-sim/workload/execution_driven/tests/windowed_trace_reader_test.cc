/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

windowed_trace_reader_test.cc -- phase-7 §10.4 WindowedTraceReader fixture,
rewritten for the P0 turn-0 late-discovery fix (2026-08-30): index pass +
turn-0 arrival calendar (方案见 WindowedTraceReader.hh; 2问题分析与解决
方案kimi.md §4.3). Exercises the calendar reader standalone -- no network
simulation, no baseline artifacts touched:

  Part A  Calendar submission: the FIRST pump runs the
          whole index pass (structure validation, per-row metrics-order
          registration, turn>0 pre-registration) and submits every turn-0
          in ARRIVAL order; consumption shrinks the outstanding set.
  Part B  Row semantics: turn-0 rows are submitted exactly once (command
          queue count == turn-0 row count), turn>0 rows are registered but
          never submitted (future_alarm path), EOF yields data_rows == total
          data rows after one pump (the A.4-removed row-window knob no
          longer exists: one pump indexes the whole file).
  Part C  Out-of-range rejection: a turn-0 arrival beyond max_arrival_ns is
          rejected (never submitted, counted) regardless of its file
          position; in-range rows still submit.
  Part D  Late-arrival clamp counters SPLIT by producer path
          (RequestIngress): an external-stream arrival whose target tick is
          already past is clamped to current+1 and counted in
          late_external_stream (never in late_static_submit).
  Part E  Request-neutral no-CSV path: an empty path constructs an
          already-EOF reader (no-op pumps, zero rows).
  Part G  Backport fix (2026-08-16, sh_2.0测试 §5.1) -- DEFAULT UNBOUNDED
          arrival window: synthetic envelopes at arbitrary (well beyond
          30 s) arrivals are ALL accepted under the 0 default (constructor
          default and CLI default); no csv-derived input is involved.
  Part H  Backport fix cont. (unified 2026-08-20 中-3) -- explicit window:
          drops visible (counter) and the reader still reaches EOF,
          audit_completion() fail-closes every bad combination.
  Part I  Queue-index lifetime: turn-0 rows retain no map entry; every
          turn>0 row is pre-registered by the index pass and erased before
          its future alarm is scheduled; missing, replayed, injected, and
          duplicate lookups fail closed.
  Part J  Out-of-order consumption keeps an exact outstanding set
          (submitted-but-unfired turn-0 rows only).
  Part K  P0 fix core: out-of-order turn-0 (file order non-monotonic) --
          the fire order is the CALENDAR order (arrival, queue_index), all
          discovered at tick 0, late_static_submit == 0, arrival gate ok.
  Part L  First session block longer than the old default window (>128
          turns): every row is reachable in one pump (the old bounded
          reader could not read the whole block).
  Part M  arrival==0 boundary: t0_boundary_clamp classification, exempt
          from the gate (late_static_submit stays 0, gate ok).
  Part N  Same tick, multiple sessions: the equal-arrival group fires in
          queue_index (row) order -- the equivalence argument's core.
  Part O  Provenance sidecar gate, ONE negative case per compared arm
          (non-JSON, schema, every expect_u64 field, the contiguity flag,
          plus the missing-field shape): each tampered sidecar is the
          MATCHED record with only its target field rewritten, so the
          fail-closed exit BEFORE any Submit (zero commands enqueued) is
          attributable to that arm -- a sidecar that also zeroes
          csv_fnv1a64/csv_bytes fatal-s at the FIRST compared arm
          (csv_fnv1a64) and never reaches the arm under test.
  Part P  Session block not contiguous / turn_index gap: fail-closed exit.
  Part Q  V4 gate self-test (M5 update 2026-09-30): a backpressure-parked
          submission is a capacity artifact (late_static_submit_backpressure,
          gate PASSES); true late discovery still trips late_static_submit
          and FAILS the gate; the normal path gate passes.
  Part R  Turn>0 runtime chain (review 2026-09-26 M9-2): a commit-leg
          future arrival whose target tick is already past is clamped to
          current+1 and counted in late_future_alarm_rounding, and the
          production arrival hook (main_online.cc) drives the consumption
          leg -- every fired alarm (turn-0 and turn>0) retires its row via
          reader.notify_consumed.

Build: the CMake target AstraSim_Analytical_Congestion_Aware_WindowedReaderTest.
Run (from template/astra-sim-face):
  build/astra_analytical/build_congestion_aware/bin/\
    AstraSim_Analytical_Congestion_Aware_WindowedReaderTest
*******************************************************************************/

#include <cassert>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "astra-sim/workload/execution_driven/DecisionMailbox.hh"
#include "astra-sim/workload/execution_driven/OnlineCli.hh"
#include "astra-sim/workload/execution_driven/RequestIngress.hh"
#include "astra-sim/workload/execution_driven/ServiceCoordinator.hh"
#include "astra-sim/workload/execution_driven/WindowedTraceReader.hh"
#include "astra-sim/workload/execution_driven/tests/fixture_provenance_sidecar.hh"
#include "common/EventQueue.h"

using namespace AstraSim::ExecutionDriven;
using namespace NetworkAnalytical;

namespace {

static std::string fixture_path(const char* base, const char* ext = ".csv") {
    // Parallel-reader validation (phase 5-6): per-process fixture paths so
    // concurrent instances do not race on shared /tmp files (fixture-only;
    // same fix as the sh_3.0 validation round).
    return std::string("/tmp/") + base + "_" + std::to_string(::getpid()) + ext;
}


bool g_ok = true;

void expect(const bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "[windowed_reader_test] FAIL: %s\n", what);
        g_ok = false;
    }
}

template <typename Fn>
void expect_sigabrt(Fn&& fn, const char* what) {
    const pid_t pid = ::fork();
    if (pid < 0) {
        expect(false, "fork failed in fail-closed assertion");
        return;
    }
    if (pid == 0) {
        std::freopen("/dev/null", "w", stderr);
        fn();
        ::_exit(0);
    }
    int status = 0;
    const pid_t waited = ::waitpid(pid, &status, 0);
    expect(waited == pid && WIFSIGNALED(status) &&
               WTERMSIG(status) == SIGABRT,
           what);
}

template <typename Fn>
void expect_exit_code(Fn&& fn, const int code, const char* what) {
    // Fail-closed reader_fatal path: std::exit(EXIT_FAILURE) in a forked
    // child (the [Error] line itself goes to the child's stderr).
    const pid_t pid = ::fork();
    if (pid < 0) {
        expect(false, "fork failed in fail-closed assertion");
        return;
    }
    if (pid == 0) {
        std::freopen("/dev/null", "w", stderr);
        fn();
        ::_exit(0);
    }
    int status = 0;
    const pid_t waited = ::waitpid(pid, &status, 0);
    expect(waited == pid && WIFEXITED(status) && WEXITSTATUS(status) == code,
           what);
}

// cpp-ed-tests-b-2: child-side probe for the tampered-sidecar zero-Submit
// check below. Captureless function so std::atexit accepts it;
// g_probe_ingress is assigned inside the forked child, so it points at the
// CHILD's ingress copy whose queue the failing index pass would have fed.
RequestIngress* g_probe_ingress = nullptr;

void verify_zero_submits_at_exit() {
    const bool zero_submits =
        g_probe_ingress != nullptr &&
        g_probe_ingress->pending_command_count() == 0;
    ::_exit(zero_submits ? EXIT_FAILURE : 2);
}

void write_csv(const std::string& path,
               const std::vector<std::string>& rows) {
    // P1 fix (2026-09-28, ported to this repo 2026-10-07): the provenance
    // gate is fail-closed on an absent sidecar, so every queue-CSV fixture
    // ships its matching provenance record (computed exactly like the
    // materializer / the reader's index pass). Deliberately-malformed
    // fixtures abort in-row before the gate, so their sidecar values are
    // never consulted.
    FixtureProvenance::write_queue_fixture(path, rows);
}

// 10 sessions (turn-0 arrivals 1..10 s, monotonic) + session_0 turn-1;
// contiguous session blocks (session_0's turn>0 row directly after its
// turn-0 row -- the P0 reader fail-closes on non-contiguous blocks).
std::vector<std::string> sample_rows() {
    std::vector<std::string> rows;
    for (int i = 0; i < 10; ++i) {
        rows.push_back("session_" + std::to_string(i) + ",0,session_" +
                       std::to_string(i) + "_request_0,100,50," +
                       std::to_string(1000000000ULL * (i + 1)) + ",,d");
        if (i == 0) {
            rows.push_back("session_0,1,session_0_request_1,100,50,,"
                           "5000000000,d");
        }
    }
    return rows;
}

void test_calendar_submission() {
    const std::string csv = fixture_path("windowed_reader_test_window");
    write_csv(csv, sample_rows());

    EventQueue eq;
    DecisionMailbox mailbox;
    ServiceCoordinator svc;
    RequestIngress ingress;
    ingress.bind(&eq, &mailbox, &svc);

    WindowedTraceReader reader(csv, ingress);
    // First pump: the WHOLE file is indexed (window no longer bounds
    // discovery) and every turn-0 row is submitted in arrival order.
    expect(!reader.pump(), "first pump indexes and drains the calendar");
    expect(reader.eof(), "EOF after the first pump (calendar fully drained)");
    expect(reader.rows_read() == 11,
           "index pass reads all 11 rows regardless of window");
    expect(reader.data_rows() == 11, "data_rows == 11 after one pump");
    expect(reader.turn0_data_rows() == 10, "10 turn-0 rows");
    expect(ingress.pending_command_count() == 10,
           "all 10 turn-0 Submits queued by the first pump");
    expect(reader.current_window_occupancy() == 10,
           "outstanding = submitted-but-unfired turn-0 rows");

    // Later pumps are no-ops (nothing left to index or submit).
    expect(!reader.pump(), "later pump returns false at EOF");
    expect(reader.rows_read() == 11, "later pump reads nothing");

    // Consumption shrinks the outstanding set (turn-0 alarm fired). Row 1
    // is session_0's turn>0 row (never outstanding); retire two turn-0
    // rows out of order: 0 (session_0) and 2 (session_1).
    reader.notify_consumed(0);
    expect(reader.current_window_occupancy() == 9,
           "consumption retires one outstanding turn-0 row");
    reader.notify_consumed(2);
    expect(reader.current_window_occupancy() == 8,
           "out-of-order consumption retires exactly itself");
    expect(reader.rejected_out_of_range() == 0, "no rejections in part A");
    std::remove(csv.c_str());
    std::printf("[fixture] part A PASS: calendar submission\n");
}

void test_row_semantics() {
    const std::string csv = fixture_path("windowed_reader_test_semantics");
    write_csv(csv, sample_rows());

    EventQueue eq;
    DecisionMailbox mailbox;
    ServiceCoordinator svc;
    RequestIngress ingress;
    ingress.bind(&eq, &mailbox, &svc);

    // One pump indexes the whole file and submits every turn-0 in arrival
    // order (whole-file index pass; no row window anymore).
    WindowedTraceReader reader(csv, ingress);
    expect(!reader.pump(), "one pump drains the calendar");
    expect(reader.eof(), "EOF after the single pump");
    expect(reader.data_rows() == 11, "data_rows == 11");
    expect(reader.rows_read() == 11, "rows_read == 11");
    expect(ingress.pending_command_count() == 10,
           "exactly the 10 turn-0 rows are submitted; turn>0 never is");
    std::remove(csv.c_str());
    std::printf("[fixture] part B PASS: turn-0 Submit / turn>0 future-alarm "
                "semantics (whole-file index pass)\n");
}

// (2026-10-01) Pins for the three same-day reader gates: CRLF blank
// lines, a UTF-8 BOM header, and short (<7-column) data rows.
// P1 port (2026-10-07): every legal arm below ships a byte-accurate
// provenance sidecar so the pass/fail is attributable to the shape gate
// under test, not to the fail-closed absent-sidecar gate; the counters
// follow the reader's own line semantics (a CRLF blank line is skipped,
// not counted), with the hash over the exact raw bytes via the shared
// helper's FNV routine.
void write_raw_csv(const std::string& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary);
    out << bytes;
}

void write_raw_sidecar(const std::string& path, const std::string& bytes,
                       const uint64_t data_rows, const uint64_t sessions,
                       const uint64_t turn0_count, const uint64_t t0_min,
                       const uint64_t t0_max, const uint64_t inversions) {
    const uint64_t fnv =
        FixtureProvenance::fnv1a64_bytes(14695981039346656037ULL, bytes);
    std::ofstream out(path + ".provenance.json", std::ios::binary);
    out << "{\"schema\":1,\"generator_version\":\"test-fixture\","
        << "\"csv_sha256\":\"test-fixture (human audit only)\","
        << "\"csv_fnv1a64\":" << fnv << ",\"csv_bytes\":" << bytes.size()
        << ",\"data_rows\":" << data_rows << ",\"sessions\":" << sessions
        << ",\"turn0_count\":" << turn0_count
        << ",\"turn0_arrival_min_ns\":" << t0_min
        << ",\"turn0_arrival_max_ns\":" << t0_max
        << ",\"turn0_adjacent_inversions\":" << inversions
        << ",\"session_blocks_contiguous\":true}\n";
}

void test_reader_input_shape_gates() {
    // (1) CRLF blank lines are skipped exactly like LF blank lines.
    {
        const std::string csv =
            fixture_path("windowed_reader_test_crlf_blank");
        const std::string bytes =
            std::string(kRequestQueueCsvHeader) +
            "\r\n"
            "\r\n"  // CRLF blank line -- used to reach the header
                    // branch as "\r" and die with a wrong message
            "s0,0,s0_r0,100,10,500,,d\r\n"
            "\r\n"  // a second one among data rows
            "s0,1,s0_r1,100,10,,1000,d\r\n";
        write_raw_csv(csv, bytes);
        // Reader line semantics: both "\r" blank lines are skipped, so
        // data_rows=2 / one session / one turn-0 (arrival 500); the
        // hash/byte counters stay on the raw CRLF bytes.
        write_raw_sidecar(csv, bytes,
                          /*data_rows=*/2, /*sessions=*/1,
                          /*turn0_count=*/1, /*t0_min=*/500,
                          /*t0_max=*/500, /*inversions=*/0);
        EventQueue eq;
        DecisionMailbox mailbox;
        ServiceCoordinator svc;
        RequestIngress ingress;
        ingress.bind(&eq, &mailbox, &svc);
        WindowedTraceReader reader(csv, ingress);
        expect(!reader.pump(), "CRLF blank lines: one pump drains");
        expect(reader.data_rows() == 2,
               "CRLF blank lines are not data rows");
        expect(ingress.pending_command_count() == 1,
               "CRLF blank lines: only the turn-0 row is submitted");
        std::remove(csv.c_str());
        std::remove((csv + ".provenance.json").c_str());
    }
    // (2) A UTF-8 BOM before the header is accepted (utf-8-sig parity).
    {
        const std::string csv = fixture_path("windowed_reader_test_bom");
        const std::string bytes =
            std::string("\xEF\xBB\xBF") + kRequestQueueCsvHeader +
            "\n"
            "s0,0,s0_r0,100,10,500,,d\n";
        write_raw_csv(csv, bytes);
        // First non-empty line = the BOM-prefixed header; one data row,
        // one turn-0 (arrival 500); hash over the raw bytes incl. BOM.
        write_raw_sidecar(csv, bytes,
                          /*data_rows=*/1, /*sessions=*/1,
                          /*turn0_count=*/1, /*t0_min=*/500,
                          /*t0_max=*/500, /*inversions=*/0);
        EventQueue eq;
        DecisionMailbox mailbox;
        ServiceCoordinator svc;
        RequestIngress ingress;
        ingress.bind(&eq, &mailbox, &svc);
        WindowedTraceReader reader(csv, ingress);
        expect(!reader.pump(), "BOM header: one pump drains");
        expect(reader.data_rows() == 1, "BOM header: the data row counts");
        expect(ingress.pending_command_count() == 1,
               "BOM header: the turn-0 row is submitted");
        std::remove(csv.c_str());
        std::remove((csv + ".provenance.json").c_str());
    }
    // (3) A short row (fewer than 7 columns) fails closed at the reader
    // (reader_fatal exits EXIT_FAILURE; 4 columns: valid-looking turn-0
    // shape, but the interval column does not exist -- pre-fix it
    // registered with interval=0).
    {
        const std::string csv = fixture_path("windowed_reader_test_short_row");
        const std::string bytes =
            std::string(kRequestQueueCsvHeader) +
            "\n"
            "s0,0,s0_r0,100,10\n"
            "s0,1,s0_r1,100,10\n";
        write_raw_csv(csv, bytes);
        // Sidecar ships for failure attribution (the row check must fire,
        // not the absent-sidecar gate); its values are never consulted --
        // the reader fatal-s in-row during the index pass, BEFORE the
        // gate. Helper-rule stats over the bytes: two data rows, no
        // arrival field (taken<6), so zero turn-0.
        write_raw_sidecar(csv, bytes,
                          /*data_rows=*/2, /*sessions=*/1,
                          /*turn0_count=*/0, /*t0_min=*/0,
                          /*t0_max=*/0, /*inversions=*/0);
        expect_exit_code(
            [&csv]() {
                EventQueue eq;
                DecisionMailbox mailbox;
                ServiceCoordinator svc;
                RequestIngress ingress;
                ingress.bind(&eq, &mailbox, &svc);
                WindowedTraceReader reader(csv, ingress);
                reader.pump();
            },
            EXIT_FAILURE, "reader aborts a <7-column data row");
        std::remove(csv.c_str());
        std::remove((csv + ".provenance.json").c_str());
    }
    std::printf("[fixture] part S PASS: CRLF blank / BOM header / short-row "
                "gates\n");
}

void test_out_of_range_rejection() {
    const std::string csv = fixture_path("windowed_reader_test_reject");
    // The out-of-range row sits FIRST in the file (data row 0); the calendar
    // submits in arrival order and processes the rejection per entry either
    // way.
    write_csv(csv, {
        "session_0,0,session_0_request_0,100,50,40000000000,,d",
        "session_1,0,session_1_request_0,100,50,20000000000,,d",
    });

    EventQueue eq;
    DecisionMailbox mailbox;
    ServiceCoordinator svc;
    RequestIngress ingress;
    ingress.bind(&eq, &mailbox, &svc);

    // max_arrival_ns = 30s: data row 0 (40s, written first) is rejected,
    // data row 1 (20s) is in range and submitted.
    WindowedTraceReader reader(csv, ingress,
                               /*max_arrival_ns=*/30000000000ULL);
    reader.pump();
    expect(reader.rejected_out_of_range() == 1,
           "out-of-range turn-0 row is rejected and counted");
    expect(reader.rows_read() == 2, "rejected row still counted as read");
    expect(ingress.pending_command_count() == 1,
           "only the in-range row is submitted");
    expect(reader.eof(), "reader reached EOF despite the rejection");
    std::remove(csv.c_str());
    std::printf("[fixture] part C PASS: simulation-out-of-range rejection\n");
}

void test_late_counter_split() {
    EventQueue eq;
    DecisionMailbox mailbox;
    ServiceCoordinator svc;
    RequestIngress ingress;
    ingress.bind(&eq, &mailbox, &svc);

    // First: an in-time external-stream arrival (alarm at 1000 > current 0)
    // is NOT counted anywhere.
    IngressCommand cmd;
    cmd.kind = IngressCommandKind::Submit;
    cmd.envelope.session_id = "s0";
    cmd.envelope.turn_index = 0;
    cmd.envelope.request_id = "s0_r0";
    cmd.envelope.prefill_length = 100;
    cmd.envelope.decode_length = 50;
    cmd.envelope.arrival_world_ns = 1000;
    expect(ingress.enqueue_command(cmd), "Submit accepted");
    ingress.drain_commands();
    expect(ingress.late_arrival_count() == 0,
           "in-time arrival is not counted as late");
    expect(ingress.late_external_stream_count() == 0,
           "in-time arrival: external-stream counter stays 0");
    expect(ingress.late_static_submit_count() == 0,
           "in-time arrival: static counter stays 0");

    // Advance the queue to t=1000 (the alarm fires; harmless without a
    // hook), then submit a request whose target tick (1000) is already
    // past: the drain clamps it to current+1 and counts ONE late external
    // stream arrival (the fixture's default source), never static.
    while (!eq.finished() && eq.get_current_time() < 1000) {
        eq.proceed();
    }
    expect(eq.get_current_time() == 1000, "queue advanced to target tick");
    IngressCommand cmd2;
    cmd2.kind = IngressCommandKind::Submit;
    cmd2.envelope.session_id = "s1";
    cmd2.envelope.turn_index = 0;
    cmd2.envelope.request_id = "s1_r0";
    cmd2.envelope.prefill_length = 100;
    cmd2.envelope.decode_length = 50;
    cmd2.envelope.arrival_world_ns = 1000;
    expect(ingress.enqueue_command(cmd2), "late Submit accepted");
    ingress.drain_commands();
    expect(ingress.late_external_stream_count() == 1,
           "late external-stream clamp counted exactly once");
    expect(ingress.late_static_submit_count() == 0,
           "external-stream lateness never touches the static counter");
    expect(ingress.late_arrival_count() == 1, "total = sum of the split");

    // A future arrival is NOT counted.
    IngressCommand cmd3;
    cmd3.kind = IngressCommandKind::Submit;
    cmd3.envelope.session_id = "s2";
    cmd3.envelope.turn_index = 0;
    cmd3.envelope.request_id = "s2_r0";
    cmd3.envelope.prefill_length = 100;
    cmd3.envelope.decode_length = 50;
    cmd3.envelope.arrival_world_ns = eq.get_current_time() + 5000;
    expect(ingress.enqueue_command(cmd3), "third Submit accepted");
    ingress.drain_commands();
    expect(ingress.late_arrival_count() == 1,
           "future arrival is not counted as late");
    std::printf("[fixture] part D PASS: late clamp counters split by "
                "source\n");
}

void test_no_csv_path() {
    EventQueue eq;
    DecisionMailbox mailbox;
    ServiceCoordinator svc;
    RequestIngress ingress;
    ingress.bind(&eq, &mailbox, &svc);

    WindowedTraceReader reader("", ingress);
    expect(reader.eof(), "empty path constructs an already-EOF reader");
    expect(!reader.pump(), "pump on the no-CSV reader returns false");
    expect(reader.data_rows() == 0, "no-CSV reader reads zero rows");
    expect(ingress.pending_command_count() == 0,
           "no-CSV reader submits nothing");
    std::printf("[fixture] part E PASS: request-neutral no-CSV path\n");
}

// Backport fix (2026-08-16, sh_2.0测试 §5.1): the DEFAULT arrival window is
// UNBOUNDED. Synthetic envelopes at arbitrary arrivals (far beyond the old
// 30 s default; no materialized csv involved) must ALL be accepted.
void test_default_unbounded_window() {
    const std::string csv = fixture_path("windowed_reader_test_unbounded");
    // 3 sessions (turn-0 at 20 s / 40 s / 180 s -- the 180 s row is exactly
    // the 3-minute comparison window that exposed the defect) + 1 turn>0 row
    // in session_0's block (contiguous).
    write_csv(csv, {
        "session_0,0,session_0_request_0,100,50,20000000000,,d",
        "session_0,1,session_0_request_1,100,50,,5000000000,d",
        "session_1,0,session_1_request_0,100,50,40000000000,,d",
        "session_2,0,session_2_request_0,100,50,180000000000,,d",
    });

    // Constructor default (max_arrival_ns omitted): unbounded.
    {
        EventQueue eq;
        DecisionMailbox mailbox;
        ServiceCoordinator svc;
        RequestIngress ingress;
        ingress.bind(&eq, &mailbox, &svc);
        WindowedTraceReader reader(csv, ingress);
        reader.pump();
        expect(reader.rejected_out_of_range() == 0,
               "default window is unbounded: nothing rejected");
        expect(ingress.pending_command_count() == 3,
               "default window accepts every turn-0 row, 180 s included");
        expect(reader.data_rows() == 4 && reader.total_data_rows() == 4,
               "all rows counted (no tail scan needed at EOF)");
    }

    // CLI default (parse_online_cli with no --request-max-arrival-ns): 0.
    {
        char arg0[] = "fake_online_bin";
        char arg1[] = "--online-mode=strategy";
        char* argv[] = {arg0, arg1};
        OnlineCliOptions cli;
        std::string err;
        expect(parse_online_cli(2, argv, cli, err),
               "minimal CLI parses (unbounded-default check)");
        expect(cli.request_max_arrival_ns == 0,
               "CLI default --request-max-arrival-ns == 0 (unbounded)");
    }
    std::remove(csv.c_str());
    std::printf("[fixture] part G PASS: default unbounded arrival window "
                "(constructor + CLI defaults)\n");
}

// Backport fix cont. (unified 2026-08-20, 中-3): explicit small window ->
// drops VISIBLE (counter) and the reader still reaches EOF naturally, so
// data_rows()/total_data_rows() ARE the whole-file denominators;
// audit_completion() fail-closes every dropping/unbalanced/incomplete
// combination.
void test_explicit_window_fail_closed() {
    const std::string csv = fixture_path("windowed_reader_test_failclosed");
    // 5 sessions (turn-0 arrivals 10 s .. 90 s; the 25 s window admits only
    // the 10 s row: 4 drops) each with a turn>0 row in-block: total = 10,
    // turn0 = 5.
    write_csv(csv, {
        "session_0,0,session_0_request_0,100,50,10000000000,,d",
        "session_0,1,session_0_request_1,100,50,,5000000000,d",
        "session_1,0,session_1_request_0,100,50,30000000000,,d",
        "session_1,1,session_1_request_1,100,50,,5000000000,d",
        "session_2,0,session_2_request_0,100,50,50000000000,,d",
        "session_2,1,session_2_request_1,100,50,,5000000000,d",
        "session_3,0,session_3_request_0,100,50,70000000000,,d",
        "session_3,1,session_3_request_1,100,50,,5000000000,d",
        "session_4,0,session_4_request_0,100,50,90000000000,,d",
        "session_4,1,session_4_request_1,100,50,,5000000000,d",
    });

    EventQueue eq;
    DecisionMailbox mailbox;
    ServiceCoordinator svc;
    RequestIngress ingress;
    ingress.bind(&eq, &mailbox, &svc);
    WindowedTraceReader reader(csv, ingress,
                               /*max_arrival_ns=*/25000000000ULL);
    reader.pump();
    ingress.drain_commands();  // effective-arrival records exist from here
    expect(reader.eof(),
           "consume-at-reject: EOF reached naturally, no tail scan");
    expect(reader.rows_read() == 10,
           "the whole file was read: rejects never stalled the reader");
    expect(reader.rejected_out_of_range() == 4,
           "explicit small window rejects visibly (4 of 5 turn-0 rows)");
    expect(reader.total_data_rows() == 10,
           "whole-file denominator == data_rows at EOF");
    expect(reader.turn0_data_rows() == 5,
           "turn-0 whole-file count");
    expect(ingress.pending_command_count() == 0,
           "the single in-range Submit was drained (queue now empty)");
    // The rejected rows never entered the simulation: the arrival audit
    // excludes them by design; the one in-range row (queue 0, 10 s) is the
    // only audited arrival.
    const auto audit = reader.arrival_audit();
    expect(audit.size() == 1,
           "arrival audit covers submitted turn-0 rows only");
    expect(audit[0].queue_index == 0 && audit[0].declared_arrival_ns ==
               10000000000ULL,
           "the audited arrival is the in-range 10 s row");
    expect(reader.audit_static_arrivals().gate_ok,
           "rejections are covered by the completion audit, not the "
           "arrival gate");

    // The run-end audit arithmetic (audit_completion). Field order:
    // {total, turn0, accepted, completed, dropped}. Official-run shape
    // (20.csv 30s input): accepted counts ONLY turn-0 submissions (112);
    // turn>0 requests are covered by completed vs total.
    expect(audit_completion({1177, 112, 112, 1177, 0}) ==
               CompletionAuditVerdict::Ok,
           "audit: official-run shape passes (accepted==turn0, "
           "completed==total)");
    expect(audit_completion({7, 5, 5, 7, 0}) == CompletionAuditVerdict::Ok,
           "audit: clean synthetic run passes");
    expect(audit_completion({2091, 136, 112, 1830, 261}) ==
               CompletionAuditVerdict::Dropped,
           "audit: the sh_2.0 strategy-20 shape (2091 input, 1830 "
           "completed, 261 dropped, old audit PASSED) is Dropped");
    // An unbalanced turn-0 ledger with no recorded drop: AccountMismatch
    // (e.g. Submit commands lost to an ingress overflow).
    expect(audit_completion({7, 5, 4, 7, 0}) ==
               CompletionAuditVerdict::AccountMismatch,
           "audit: accepted + dropped != turn0 -> AccountMismatch");
    // Requests missing from the run (never-read tail rows behind a stall /
    // un-fired future alarms / unfinished requests): Incomplete.
    expect(audit_completion({7, 5, 5, 6, 0}) ==
               CompletionAuditVerdict::Incomplete,
           "audit: completed + dropped != total -> Incomplete");
    // The unified fixture's own shape: 4 drops, only the in-range row
    // accepted+completed, the 4 turn>0 orphans never scheduled -> both the
    // Dropped-first ordering and the completed shortfall must fail-closed.
    expect(audit_completion({10, 5, 1, 1, 4}) == CompletionAuditVerdict::Dropped,
           "audit: this fixture's run-end shape is Dropped (drops first)");
    std::remove(csv.c_str());
    std::printf("[fixture] part H PASS: explicit window drops visible + "
                "fail-closed completion audit\n");
}

void test_queue_index_lifetime_and_fail_closed() {
    const std::string csv = fixture_path("windowed_reader_test_queue_index");
    write_csv(csv, {
        "session_0,0,session_0_request_0,100,50,10,,d",
        "session_0,1,session_0_request_1,100,50,,90,d",
        "session_1,0,session_1_request_0,100,50,110,,d",
        "session_1,1,session_1_request_1,100,50,,90,d",
    });

    EventQueue eq;
    DecisionMailbox mailbox;
    ServiceCoordinator svc;
    RequestIngress ingress;
    ingress.bind(&eq, &mailbox, &svc);
    WindowedTraceReader reader(csv, ingress);

    // The index pass pre-registers EVERY turn>0 row up front (P0 fix:
    // O(rows), no longer bounded by the window).
    expect(reader.pump() == false, "one pump indexes and drains the file");
    expect(reader.rows_read() == 4, "all four rows indexed in one pump");
    expect(ingress.pending_command_count() == 2,
           "both turn-0 rows submitted");

    RequestEnvelope future0;
    future0.session_id = "session_0";
    future0.turn_index = 1;
    future0.request_id = "session_0_request_1";
    future0.arrival_world_ns = 100;
    ingress.schedule_future_arrival(future0);
    RequestEnvelope future1;
    future1.session_id = "session_1";
    future1.turn_index = 1;
    future1.request_id = "session_1_request_1";
    future1.arrival_world_ns = 200;
    ingress.schedule_future_arrival(future1);
    while (!eq.finished()) {
        eq.proceed();
    }

    // Death mechanism note: unlike the duplicate-registration case below
    // (fprintf + std::abort in the register path), a missing/mismatched
    // lookup dies inside schedule_future_arrival as a thrown
    // std::runtime_error -- uncaught in the forked child -> terminate ->
    // SIGABRT. The observed signal is identical either way.
    expect_sigabrt(
        [&]() {
            RequestEnvelope missing;
            missing.request_id = "missing_request";
            missing.arrival_world_ns = 300;
            ingress.schedule_future_arrival(missing);
        },
        "queue-index: missing production lookup throws runtime_error "
        "(uncaught -> SIGABRT)");
    expect_sigabrt(
        [&]() {
            RequestEnvelope injected;
            injected.request_id = "missing_request";
            injected.queue_index = 999;
            injected.arrival_world_ns = 300;
            ingress.schedule_future_arrival(injected);
        },
        "queue-index: supplied index cannot bypass missing lookup");
    expect_sigabrt(
        [&]() {
            ingress.register_queue_index("duplicate_request", 10);
            ingress.register_queue_index("duplicate_request", 11);
        },
        "queue-index: duplicate registration aborts");

    std::remove(csv.c_str());
    std::printf("[fixture] part I PASS: queue-index full pre-registration / "
                "one-shot erase / fail-closed adversarial cases\n");
}

void test_out_of_order_consumption_bound() {
    const std::string csv =
        fixture_path("windowed_reader_test_out_of_order");
    std::vector<std::string> rows;
    for (int i = 0; i < 8; ++i) {
        rows.push_back("session_" + std::to_string(i) + ",0,request_" +
                       std::to_string(i) + ",10,1," +
                       std::to_string(i + 1) + ",,d");
    }
    write_csv(csv, rows);

    EventQueue eq;
    DecisionMailbox mailbox;
    ServiceCoordinator svc;
    RequestIngress ingress;
    ingress.bind(&eq, &mailbox, &svc);
    WindowedTraceReader reader(csv, ingress);
    expect(!reader.pump() && reader.rows_read() == 8,
           "out-of-order: one pump reads and submits all eight rows");

    // Row 3 completes while rows 0..2,4..7 remain outstanding. Exactly one
    // entry leaves the outstanding set.
    reader.notify_consumed(3);
    expect(reader.current_window_occupancy() == 7,
           "out-of-order: later completion retires only itself");
    for (int64_t index = 0; index < 8; ++index) {
        reader.notify_consumed(index);
    }
    expect(reader.current_window_occupancy() == 0,
           "out-of-order: all rows eventually retired exactly once");
    expect(reader.eof() && reader.rows_read() == 8,
           "out-of-order: every row read exactly once");
    std::remove(csv.c_str());
    std::printf("[fixture] part J PASS: exact out-of-order outstanding "
                "set\n");
}

// Captures the (tick, queue_index) arrival-fire sequence of a fully drained
// calendar through the real drain + EventQueue path.
struct FireRecord {
    EventTime tick;
    int64_t queue_index;
};

std::vector<FireRecord> run_calendar_and_capture(const std::string& csv,
                                                 RequestIngress& ingress,
                                                 EventQueue& eq,
                                                 WindowedTraceReader& reader) {
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

// Part K: the core P0 scenario -- turn-0 arrivals NON-monotonic in file
// order (three session blocks: file order arrivals 500, 100, 300). The old
// row-window reader submitted in file order; the calendar must submit in
// arrival order with zero late static submits and all discovered at tick 0.
void test_calendar_out_of_order_turn0() {
    const std::string csv = fixture_path("windowed_reader_test_calendar");
    write_csv(csv, {
        "session_0,0,session_0_request_0,100,50,500,,d",
        "session_0,1,session_0_request_1,100,50,,90,d",
        "session_1,0,session_1_request_0,100,50,100,,d",
        "session_2,0,session_2_request_0,100,50,300,,d",
    });

    EventQueue eq;
    DecisionMailbox mailbox;
    ServiceCoordinator svc;
    RequestIngress ingress;
    ingress.bind(&eq, &mailbox, &svc);
    WindowedTraceReader reader(csv, ingress);
    const std::vector<FireRecord> seq =
        run_calendar_and_capture(csv, ingress, eq, reader);

    expect(seq.size() == 3, "calendar fires all three turn-0 arrivals");
    // Calendar order: 100 (queue 2), 300 (queue 3), 500 (queue 1).
    expect(seq[0].tick == 100 && seq[0].queue_index == 2,
           "earliest turn-0 (file position 3) fires first");
    expect(seq[1].tick == 300 && seq[1].queue_index == 3,
           "middle turn-0 (file position 4) fires second");
    expect(seq[2].tick == 500 && seq[2].queue_index == 0,
           "late-file-first row (queue 0, arrival 500) fires last");
    expect(ingress.late_static_submit_count() == 0,
           "no static-CSV Submit was clamped (late_static_submit == 0)");
    expect(reader.audit_static_arrivals().gate_ok,
           "arrival gate passes on the out-of-order input");
    const auto summary = reader.audit_static_arrivals();
    expect(summary.turn0_submitted == 3, "audit covered all three rows");
    expect(summary.delay_max_ns == 0, "every ingress_delay == 0");
    const auto audit = reader.arrival_audit();
    expect(audit.size() == 3 && audit[0].reader_discovered_tick == 0 &&
               audit[1].reader_discovered_tick == 0 &&
               audit[2].reader_discovered_tick == 0,
           "every turn-0 discovered at tick 0");
    std::remove(csv.c_str());
    std::printf("[fixture] part K PASS: out-of-order turn-0 fires in "
                "calendar order, zero late static submits\n");
}

// Part L: a first session block longer than the old default window (150
// turns). The old bounded reader could not hold
// the whole block; the calendar reader indexes everything in one pump.
void test_first_block_over_window() {
    const std::string csv = fixture_path("windowed_reader_test_longblock");
    std::vector<std::string> rows;
    // session_0: turn-0 at 1000 ns + 149 turn>0 rows.
    rows.push_back("session_0,0,session_0_request_0,100,50,1000,,d");
    for (int turn = 1; turn < 150; ++turn) {
        rows.push_back("session_0," + std::to_string(turn) + ",session_0_"
                       "request_" + std::to_string(turn) + ",100,50,,1000,d");
    }
    // A second early-arriving session AFTER the long block: file position
    // ~151, arrival earlier than session_0's turn>0 chain would produce.
    rows.push_back("session_1,0,session_1_request_0,100,50,500,,d");
    write_csv(csv, rows);

    EventQueue eq;
    DecisionMailbox mailbox;
    ServiceCoordinator svc;
    RequestIngress ingress;
    ingress.bind(&eq, &mailbox, &svc);
    WindowedTraceReader reader(csv, ingress);
    expect(!reader.pump(), "one pump drains the whole 151-row file");
    expect(reader.rows_read() == 151, "all rows indexed");
    expect(reader.data_rows() == 151, "data_rows == 151");
    expect(ingress.pending_command_count() == 2,
           "both turn-0 rows submitted (file position irrelevant)");
    std::remove(csv.c_str());
    std::printf("[fixture] part L PASS: first block > window fully "
                "reachable in one pump\n");
}

// Part M: arrival==0 boundary. The t=0 row is clamped to tick 1 by the
// EventQueue strict-future rule; that clamp is classified t0_boundary and
// exempt from the gate.
void test_arrival_zero_boundary() {
    const std::string csv = fixture_path("windowed_reader_test_t0");
    write_csv(csv, {
        "session_0,0,session_0_request_0,100,50,0,,d",
        "session_1,0,session_1_request_0,100,50,700,,d",
    });

    EventQueue eq;
    DecisionMailbox mailbox;
    ServiceCoordinator svc;
    RequestIngress ingress;
    ingress.bind(&eq, &mailbox, &svc);
    WindowedTraceReader reader(csv, ingress);
    const std::vector<FireRecord> seq =
        run_calendar_and_capture(csv, ingress, eq, reader);
    expect(seq.size() == 2, "both turn-0 rows fired");
    expect(seq[0].tick == 1 && seq[0].queue_index == 0,
           "arrival==0 row fires at tick 1 (t0 boundary clamp)");
    expect(seq[1].tick == 700 && seq[1].queue_index == 1,
           "regular row fires at its declared tick");
    expect(ingress.t0_boundary_clamp_count() == 1,
           "the t=0 clamp is counted in t0_boundary_clamp");
    expect(ingress.late_static_submit_count() == 0,
           "the t=0 clamp never counts as a late static submit");
    const auto summary = reader.audit_static_arrivals();
    expect(summary.gate_ok, "gate passes with the t0 boundary exemption");
    expect(summary.t0_boundary_clamp == 1, "summary carries t0_boundary=1");
    const auto audit = reader.arrival_audit();
    expect(std::string(audit[0].late_source) == "t0_boundary" &&
               audit[0].declared_arrival_ns == 0 &&
               audit[0].reader_discovered_tick == 0 &&
               audit[0].effective_arrival_ns == 1 &&
               audit[0].late_by_ns == 1,
           "audit row classified t0_boundary with effective=1, late_by=1");
    expect(std::string(audit[1].late_source) == "on_time",
           "regular row classified on_time");
    std::remove(csv.c_str());
    std::printf("[fixture] part M PASS: arrival==0 t0 boundary "
                "classification + gate exemption\n");
}

// Part N: same tick, multiple sessions. The equal-arrival group must fire
// in queue_index (file row) order -- the equivalence argument's core
// (EventQueue fires a shared tick's EventList in insertion order; the
// calendar inserts in queue_index order within an equal-arrival group,
// exactly like the old unbounded arm's row-order insertion).
void test_same_tick_multi_session() {
    const std::string csv = fixture_path("windowed_reader_test_sametick");
    write_csv(csv, {
        "session_0,0,session_0_request_0,100,50,500,,d",
        "session_1,0,session_1_request_0,100,50,500,,d",
        "session_2,0,session_2_request_0,100,50,500,,d",
        "session_3,0,session_3_request_0,100,50,200,,d",
    });

    EventQueue eq;
    DecisionMailbox mailbox;
    ServiceCoordinator svc;
    RequestIngress ingress;
    ingress.bind(&eq, &mailbox, &svc);
    WindowedTraceReader reader(csv, ingress);
    const std::vector<FireRecord> seq =
        run_calendar_and_capture(csv, ingress, eq, reader);
    expect(seq.size() == 4, "four turn-0 rows fired");
    expect(seq[0].tick == 200 && seq[0].queue_index == 3,
           "earlier tick group first");
    expect(seq[1].tick == 500 && seq[1].queue_index == 0 &&
               seq[2].tick == 500 && seq[2].queue_index == 1 &&
               seq[3].tick == 500 && seq[3].queue_index == 2,
           "equal-arrival group fires in queue_index (row) order");
    std::remove(csv.c_str());
    std::printf("[fixture] part N PASS: same-tick group ordered by "
                "queue_index\n");
}

// Part O: provenance sidecar tampering. A sidecar with a wrong digest (or
// wrong row count) must fail the run BEFORE any Submit: forked child exits
// non-zero and nothing was enqueued.
void test_provenance_tampering() {
    const std::string csv = fixture_path("windowed_reader_test_prov");
    write_csv(csv, {
        "session_0,0,session_0_request_0,100,50,1000,,d",
        "session_0,1,session_0_request_1,100,50,,1000,d",
        "session_1,0,session_1_request_0,100,50,500,,d",
    });
    const std::string sidecar = csv + ".provenance.json";

    // MATCHING sidecar (values taken from a first, ungated index pass):
    // status "matched" and the run proceeds. The exact text is kept in
    // matched_sidecar: every tampering case below copies it and rewrites
    // ONLY its target field, so the child's fatal is attributable to that
    // single comparison arm (a hand-written sidecar that also zeroes
    // csv_fnv1a64/csv_bytes fatal-s at the FIRST compared arm instead).
    std::string matched_sidecar;
    {
        // Hand-computed independent oracle for this exact 3-row fixture
        // (264 bytes; FNV-1a 64 over the raw bytes, offset basis
        // 14695981039346656037, prime 1099511628211): the index pass must
        // reproduce every field. data_rows=3, sessions=2, turn0_count=2,
        // arrival min/max = 500/1000, file order 1000 -> 500 = one
        // adjacent inversion, blocks contiguous.
        matched_sidecar =
            "{\"schema\":1,\"generator_version\":\"fixture\","
            "\"csv_sha256\":\"human-only\","
            "\"csv_fnv1a64\":17266060097935176346,"
            "\"csv_bytes\":264,"
            "\"data_rows\":3,"
            "\"sessions\":2,"
            "\"turn0_count\":2,"
            "\"turn0_arrival_min_ns\":500,"
            "\"turn0_arrival_max_ns\":1000,"
            "\"turn0_adjacent_inversions\":1,"
            "\"session_blocks_contiguous\":true}\n";
        std::ofstream out(sidecar);
        out << matched_sidecar;
    }
    {
        EventQueue eq;
        DecisionMailbox mailbox;
        ServiceCoordinator svc;
        RequestIngress ingress;
        ingress.bind(&eq, &mailbox, &svc);
        WindowedTraceReader reader(csv, ingress);
        reader.pump();
        // D2 (2026-09-28): provenance_sidecar_status() was removed with the
        // dead accessor; the gate's outcomes here stay observable through the
        // pending-command counts below (matched/absent both proceed, tampered
        // ones abort before any Submit).
        expect(ingress.pending_command_count() == 2,
               "provenance: matching sidecar submits both turn-0 rows");
    }

    // Tamper helpers over the matched record. with_field rewrites the value
    // token (digits or true/false) after "field":; without_field deletes
    // the whole "field":<value>, entry (a mid-JSON field only) so expect_u64
    // takes its "field missing" fatal instead of the mismatch one.
    auto sidecar_with_field = [&](const char* field, const char* value) {
        const std::string key = std::string("\"") + field + "\":";
        const std::size_t key_at = matched_sidecar.find(key);
        expect(key_at != std::string::npos,
               "provenance tamper helper: target field present");
        if (key_at == std::string::npos) {
            return matched_sidecar;
        }
        const std::size_t value_begin = key_at + key.size();
        std::size_t value_end = value_begin;
        while (value_end < matched_sidecar.size() &&
               matched_sidecar[value_end] != ',' &&
               matched_sidecar[value_end] != '}') {
            ++value_end;
        }
        return matched_sidecar.substr(0, value_begin) + value +
               matched_sidecar.substr(value_end);
    };
    auto sidecar_without_field = [&](const char* field) {
        const std::string key = std::string("\"") + field + "\":";
        const std::size_t key_at = matched_sidecar.find(key);
        expect(key_at != std::string::npos,
               "provenance tamper helper: target field present");
        const std::size_t entry_end =
            key_at == std::string::npos
                ? std::string::npos
                : matched_sidecar.find(',', key_at);
        expect(entry_end != std::string::npos,
               "provenance tamper helper: mid-JSON field entry");
        if (key_at == std::string::npos || entry_end == std::string::npos) {
            return matched_sidecar;
        }
        return matched_sidecar.substr(0, key_at) +
               matched_sidecar.substr(entry_end + 1);
    };
    // One forked-child negative run per tampered sidecar text: the child
    // must exit(EXIT_FAILURE) (reader_fatal) -- nothing else.
    auto expect_sidecar_fatals = [&](const std::string& text,
                                     const char* what) {
        std::ofstream out(sidecar);
        out << text;
        expect_exit_code(
            [&]() {
                EventQueue eq;
                DecisionMailbox mailbox;
                ServiceCoordinator svc;
                RequestIngress ingress;
                ingress.bind(&eq, &mailbox, &svc);
                WindowedTraceReader tampered(csv, ingress);
                tampered.pump();
            },
            EXIT_FAILURE, what);
    };

    // Wrong on EVERY compared field: csv_fnv1a64 is the FIRST arm the gate
    // compares, so this case exercises the digest-mismatch fatal itself;
    // the child must exit(1) with zero submissions. (The reader prints
    // [Error] to the child's stderr.)
    {
        std::ofstream out(sidecar);
        out << "{\"schema\":1,\"generator_version\":\"fixture\","
               "\"csv_sha256\":\"human-only\",\"csv_fnv1a64\":1,"
               "\"csv_bytes\":1,\"data_rows\":3,\"sessions\":2,"
               "\"turn0_count\":2,\"turn0_arrival_min_ns\":500,"
               "\"turn0_arrival_max_ns\":1000,"
               "\"turn0_adjacent_inversions\":1,"
               "\"session_blocks_contiguous\":true}\n";
    }
    {
        EventQueue eq;
        DecisionMailbox mailbox;
        ServiceCoordinator svc;
        RequestIngress ingress;
        ingress.bind(&eq, &mailbox, &svc);
        // The fail-closed index pass runs in a forked child (expect_exit_code
        // forks before invoking the lambda, so &ingress inside it is the
        // child's copy). The zero-Submit property must be observed on THAT
        // child's ingress at the moment it exits: asserting on parent-side
        // objects that never pumped the CSV was trivially true (review
        // 2026-09-26 cpp-ed-tests-b-2). The child arms an atexit probe
        // (std::exit runs atexit handlers) that re-exits EXIT_FAILURE iff
        // zero Submits were enqueued; a nonzero queue exits 2 and the parent
        // exit-code assertion below fails.
        expect_exit_code(
            [&]() {
                g_probe_ingress = &ingress;
                std::atexit(verify_zero_submits_at_exit);
                WindowedTraceReader tampered(csv, ingress);
                tampered.pump();
            },
            EXIT_FAILURE,
            "provenance: wrong digest fail-closes the run with zero "
            "Submits enqueued");
    }

    // One negative case per remaining comparison arm, each starting from
    // the matched record with ONLY the target field rewritten (gate order:
    // schema, csv_fnv1a64, csv_bytes, data_rows, sessions, turn0_count,
    // turn0_arrival_min_ns, turn0_arrival_max_ns,
    // turn0_adjacent_inversions, session_blocks_contiguous; the non-JSON
    // shape fatals before any field comparison). Review A2 (2026-09-28):
    // the previous "wrong data_rows only" case hand-wrote
    // csv_fnv1a64/csv_bytes as 0, so the child fatal-s at the csv_fnv1a64
    // arm -- the data_rows arm, and every other arm, had zero coverage.
    expect_sidecar_fatals(sidecar_with_field("schema", "2"),
                          "provenance: wrong sidecar schema fail-closes "
                          "the run");
    expect_sidecar_fatals(sidecar_with_field("csv_bytes", "1"),
                          "provenance: wrong byte count fail-closes the "
                          "run");
    expect_sidecar_fatals(sidecar_with_field("data_rows", "999"),
                          "provenance: wrong row count fail-closes the "
                          "run");
    expect_sidecar_fatals(sidecar_with_field("sessions", "999"),
                          "provenance: wrong session count fail-closes "
                          "the run");
    expect_sidecar_fatals(sidecar_with_field("turn0_count", "999"),
                          "provenance: wrong turn-0 count fail-closes the "
                          "run");
    expect_sidecar_fatals(
        sidecar_with_field("turn0_arrival_min_ns", "0"),
        "provenance: wrong turn-0 arrival minimum fail-closes the run");
    expect_sidecar_fatals(
        sidecar_with_field("turn0_arrival_max_ns", "0"),
        "provenance: wrong turn-0 arrival maximum fail-closes the run");
    // Actual inversions on this fixture are 1 (file-order turn-0 arrivals
    // 1000 then 500); claiming 0 is a mismatch.
    expect_sidecar_fatals(
        sidecar_with_field("turn0_adjacent_inversions", "0"),
        "provenance: wrong inversion count fail-closes the run");
    expect_sidecar_fatals(
        sidecar_with_field("session_blocks_contiguous", "false"),
        "provenance: sidecar claiming non-contiguous blocks fail-closes "
        "the run");
    // A missing (or non-unsigned) compared field: expect_u64's other
    // fatal, "field missing or not an unsigned integer".
    expect_sidecar_fatals(
        sidecar_without_field("sessions"),
        "provenance: sidecar missing a compared field fail-closes the "
        "run");
    // Not JSON at all: fatal before any field comparison.
    expect_sidecar_fatals("{\"schema\":1,",
                          "provenance: non-JSON sidecar fail-closes the "
                          "run");

    // Absent sidecar: FAIL-CLOSED (P1 fix 2026-09-28, ported to this repo
    // 2026-10-07; was "absent (no gate)"). A queue without its provenance
    // record cannot be told apart from a truncated one -- the run must
    // exit non-zero with zero Submits.
    std::remove(sidecar.c_str());
    {
        EventQueue eq;
        DecisionMailbox mailbox;
        ServiceCoordinator svc;
        RequestIngress ingress;
        ingress.bind(&eq, &mailbox, &svc);
        expect_exit_code(
            [&]() {
                WindowedTraceReader reader(csv, ingress);
                reader.pump();
            },
            EXIT_FAILURE,
            "provenance: absent sidecar fail-closes the run");
        expect(ingress.pending_command_count() == 0,
               "provenance: zero Submits enqueued on absent sidecar");
    }
    std::remove(csv.c_str());
    std::remove((csv + ".provenance.json").c_str());
    std::printf("[fixture] part O PASS: provenance tampering fail-closes "
                "with zero submissions\n");
}

// Part P: session block not contiguous / turn_index gap -> fail-closed exit
// before any Submit.
void test_structure_validation_fail_closed() {
    const std::string csv = fixture_path("windowed_reader_test_struct");

    // Non-contiguous: session_0 reappears after session_1's block.
    write_csv(csv, {
        "session_0,0,session_0_request_0,100,50,1000,,d",
        "session_1,0,session_1_request_0,100,50,500,,d",
        "session_0,1,session_0_request_1,100,50,,1000,d",
    });
    expect_exit_code(
        [&]() {
            EventQueue eq;
            DecisionMailbox mailbox;
            ServiceCoordinator svc;
            RequestIngress ingress;
            ingress.bind(&eq, &mailbox, &svc);
            WindowedTraceReader tampered(csv, ingress);
            tampered.pump();
        },
        EXIT_FAILURE,
        "structure: non-contiguous session block fail-closes");

    // Turn gap: session block jumps turn 0 -> turn 2.
    write_csv(csv, {
        "session_0,0,session_0_request_0,100,50,1000,,d",
        "session_0,2,session_0_request_2,100,50,,1000,d",
    });
    expect_exit_code(
        [&]() {
            EventQueue eq;
            DecisionMailbox mailbox;
            ServiceCoordinator svc;
            RequestIngress ingress;
            ingress.bind(&eq, &mailbox, &svc);
            WindowedTraceReader tampered(csv, ingress);
            tampered.pump();
        },
        EXIT_FAILURE,
        "structure: turn_index gap fail-closes");

    // turn-0 row with EMPTY arrival column: fail-closed.
    write_csv(csv, {
        "session_0,0,session_0_request_0,100,50,,,d",
    });
    expect_exit_code(
        [&]() {
            EventQueue eq;
            DecisionMailbox mailbox;
            ServiceCoordinator svc;
            RequestIngress ingress;
            ingress.bind(&eq, &mailbox, &svc);
            WindowedTraceReader tampered(csv, ingress);
            tampered.pump();
        },
        EXIT_FAILURE,
        "structure: turn-0 row with empty arrival fail-closes");

    // turn>0 row WITH an explicit arrival: fail-closed.
    write_csv(csv, {
        "session_0,0,session_0_request_0,100,50,1000,,d",
        "session_0,1,session_0_request_1,100,50,1700,1000,d",
    });
    expect_exit_code(
        [&]() {
            EventQueue eq;
            DecisionMailbox mailbox;
            ServiceCoordinator svc;
            RequestIngress ingress;
            ingress.bind(&eq, &mailbox, &svc);
            WindowedTraceReader tampered(csv, ingress);
            tampered.pump();
        },
        EXIT_FAILURE,
        "structure: turn>0 row with explicit arrival fail-closes");

    std::remove(csv.c_str());
    std::printf("[fixture] part P PASS: session/turn structure validation "
                "fail-closed\n");
}

// Part Q (V4 gate self-test, M5 update 2026-09-30, mirrored from the joint
// repo): a submission parked by ingress BACKPRESSURE (capacity 1) and
// drained after the clock advanced is a capacity artifact -- it counts into
// late_static_submit_backpressure and the gate PASSES. A row the reader
// itself discovers late (no backpressure: the first pump happens after the
// clock advanced) still trips late_static_submit and FAILS the gate. The
// normal path passes.
void noop_callback(void*) {}

void test_gate_trips_on_delayed_submission() {
    const std::string csv = fixture_path("windowed_reader_test_gate");
    write_csv(csv, {
        "session_0,0,session_0_request_0,100,50,1000,,d",
        "session_1,0,session_1_request_0,100,50,2000,,d",
    });

    EventQueue eq;
    DecisionMailbox mailbox;
    ServiceCoordinator svc;
    RequestIngress ingress(/*capacity=*/1);
    ingress.bind(&eq, &mailbox, &svc);
    WindowedTraceReader reader(csv, ingress);
    reader.pump();
    expect(ingress.pending_command_count() == 1,
           "capacity-1 ingress parks the second calendar entry");
    ingress.drain_commands();  // schedules the alarm at 1000
    // Advance the clock far past 2000 (a dummy event at 5000): the parked
    // turn-0 (declared 2000) is now submitted LATE on the next pump.
    eq.schedule_event(5000, noop_callback, nullptr);
    while (!eq.finished()) {
        eq.proceed();
    }
    expect(eq.get_current_time() == 5000, "clock advanced to 5000");
    reader.pump();  // submits the parked entry at discovered=5000
    ingress.drain_commands();
    // M5: the parked entry's clamp is a capacity artifact -- the reader
    // offered it at t=0 and the bounded queue deferred it.
    expect(ingress.late_static_submit_count() == 0,
           "backpressure-deferred Submit is NOT in late_static_submit");
    expect(ingress.late_static_submit_backpressure_count() == 1,
           "backpressure-deferred Submit counts in the M5 bucket");
    expect(ingress.t0_boundary_clamp_count() == 0,
           "not a t0 boundary (declared 2000)");
    const auto summary = reader.audit_static_arrivals();
    expect(summary.gate_ok,
           "arrival gate PASSES on the backpressure-deferred submission "
           "(M5: capacity artifact, exempt)");
    expect(summary.late_static_submit == 0, "summary late_static_submit=0");
    expect(summary.late_static_submit_backpressure == 1,
           "summary late_static_submit_backpressure=1");
    const auto audit = reader.arrival_audit();
    bool found_deferred = false;
    for (const auto& row : audit) {
        if (std::string(row.late_source) == "capacity_backpressure") {
            found_deferred = row.declared_arrival_ns == 2000 &&
                             row.reader_discovered_tick == 5000 &&
                             row.effective_arrival_ns == 5001 &&
                             row.late_by_ns == 3001 &&
                             row.backpressure_deferred;
        }
    }
    expect(found_deferred,
           "audit row records declared=2000 discovered=5000 "
           "effective=5001 late_by=3001 as capacity_backpressure");

    // True late discovery (no backpressure: the first pump happens after the
    // clock advanced past both arrivals) still trips the gate.
    {
        EventQueue eq_late;
        DecisionMailbox mailbox_late;
        ServiceCoordinator svc_late;
        RequestIngress ingress_late;
        ingress_late.bind(&eq_late, &mailbox_late, &svc_late);
        WindowedTraceReader reader_late(csv, ingress_late);
        eq_late.schedule_event(5000, noop_callback, nullptr);
        while (!eq_late.finished()) {
            eq_late.proceed();
        }
        reader_late.pump();  // first discovery at 5000: no park, pure lateness
        ingress_late.drain_commands();
        expect(ingress_late.late_static_submit_count() == 2,
               "true late discovery counts in late_static_submit");
        expect(ingress_late.late_static_submit_backpressure_count() == 0,
               "true late discovery is not a backpressure artifact");
        expect(!reader_late.audit_static_arrivals().gate_ok,
               "arrival gate still FAILS on true late discovery");
    }

    // Normal path (fresh ingress, no backpressure): gate passes.
    {
        EventQueue eq2;
        DecisionMailbox mailbox2;
        ServiceCoordinator svc2;
        RequestIngress ingress2;
        ingress2.bind(&eq2, &mailbox2, &svc2);
        // No third argument: default max_arrival_ns=0 (UNBOUNDED), so the
        // arrivals 1000/2000 submit normally and the gate assertions below
        // run against real (nonzero) submissions.
        WindowedTraceReader reader2(csv, ingress2);
        reader2.pump();
        ingress2.drain_commands();
        while (!eq2.finished()) {
            eq2.proceed();
        }
        expect(ingress2.late_static_submit_count() == 0,
               "normal path: no late static submits");
        expect(reader2.audit_static_arrivals().gate_ok,
               "normal path: arrival gate passes");
    }
    std::remove(csv.c_str());
    std::printf("[fixture] part Q PASS: gate exempts backpressure artifacts, "
                "trips true late discovery, passes on the normal path\n");
}

// Part R (review 2026-09-26 M9-2): the turn>0 runtime chain no other part
// reaches end to end. (a) The late-future-alarm clamp: schedule_future_arrival
// with a target tick at or before the current clock clamps the alarm to
// current+1 and counts late_future_alarm_rounding (RequestIngress.cc), the
// branch every prior fixture skipped (all future alarms were strictly in
// schedule-time future). (b) The consumption leg: the production arrival hook
// (main_online.cc) calls reader.notify_consumed(env.queue_index) inside
// arrival_cb, so each FIRED alarm retires its row from the reader's
// outstanding set -- previously notify_consumed was only ever driven by hand.
void test_future_arrival_clamp_and_consumption_leg() {
    const std::string csv =
        fixture_path("windowed_reader_test_future_clamp");
    write_csv(csv, {
        "session_0,0,session_0_request_0,100,50,100,,d",
        "session_0,1,session_0_request_1,100,50,,90,d",
    });

    EventQueue eq;
    DecisionMailbox mailbox;
    ServiceCoordinator svc;
    RequestIngress ingress;
    ingress.bind(&eq, &mailbox, &svc);
    WindowedTraceReader reader(csv, ingress);

    // Production consumption leg: the arrival hook is the main_online.cc
    // shape (notify the reader from inside arrival_cb), extended with a fire
    // log and occupancy snapshots for the assertions below.
    std::vector<FireRecord> seq;
    std::vector<size_t> occupancy_after_notify;
    ingress.set_arrival_hook([&](const RequestEnvelope& env) {
        seq.push_back(FireRecord{eq.get_current_time(), env.queue_index});
        reader.notify_consumed(env.queue_index);
        occupancy_after_notify.push_back(reader.current_window_occupancy());
    });

    // Index pass: one turn-0 Submit (arrival 100); the turn>0 row is
    // pre-registered (one-shot queue-index lookup for request
    // session_0_request_1).
    expect(!reader.pump(), "future-clamp: one pump indexes the file");
    expect(reader.rows_read() == 2, "future-clamp: both rows indexed");
    expect(ingress.pending_command_count() == 1,
           "future-clamp: only the turn-0 row submitted");
    ingress.drain_commands();  // arms the turn-0 alarm at 100
    expect(reader.current_window_occupancy() == 1,
           "future-clamp: the turn-0 row is outstanding before its alarm");

    // Turn-0 alarm fires at 100; the hook's notify_consumed retires the row.
    while (!eq.finished()) {
        eq.proceed();
    }
    expect(eq.get_current_time() == 100, "future-clamp: turn-0 fired at 100");
    expect(reader.current_window_occupancy() == 0,
           "future-clamp: hook->notify_consumed retired the turn-0 row");

    // Commit-leg stand-in (committer Part P covers commit->schedule; here the
    // RUNTIME branch is the target): the turn>0 arrival is scheduled with the
    // interval-based target 100+90=190 while the clock is already at 5000 --
    // the late-future-alarm rounding case. The one-shot lookup is consumed
    // and the alarm clamps to current+1.
    eq.schedule_event(5000, noop_callback, nullptr);
    while (!eq.finished()) {
        eq.proceed();
    }
    expect(eq.get_current_time() == 5000,
           "future-clamp: clock advanced past the interval-based target");
    RequestEnvelope future;
    future.session_id = "session_0";
    future.turn_index = 1;
    future.request_id = "session_0_request_1";
    future.prefill_length = 100;
    future.decode_length = 50;
    future.arrival_world_ns = 190;
    ingress.schedule_future_arrival(future);
    expect(ingress.late_future_alarm_rounding_count() == 1,
           "future-clamp: past-target future arrival counted exactly once "
           "in late_future_alarm_rounding");
    expect(ingress.late_static_submit_count() == 0 &&
               ingress.late_external_stream_count() == 0,
           "future-clamp: the clamp stays out of the static/external "
           "counters");
    expect(eq.get_current_time() == 5000 && !eq.finished(),
           "future-clamp: the clamped alarm waits at current+1");

    while (!eq.finished()) {
        eq.proceed();
    }
    expect(seq.size() == 2, "future-clamp: both arrivals fired");
    expect(seq[0].tick == 100 && seq[0].queue_index == 0,
           "future-clamp: turn-0 fired at its declared tick");
    expect(seq[1].tick == 5001 && seq[1].queue_index == 1,
           "future-clamp: past-target future alarm clamped to current+1 "
           "(5001) with the resolved queue index");
    expect(occupancy_after_notify == std::vector<size_t>({0, 0}),
           "future-clamp: the consumption leg kept the outstanding set "
           "exact through both fired alarms");
    expect(reader.audit_static_arrivals().gate_ok,
           "future-clamp: the future-alarm rounding is never gated");

    std::remove(csv.c_str());
    std::printf("[fixture] part R PASS: late future-alarm clamp to "
                "current+1 + hook->notify_consumed consumption leg\n");
}

}  // namespace

int main(int /*argc*/, char* /*argv*/[]) {
    test_calendar_submission();
    test_row_semantics();
    test_reader_input_shape_gates();
    test_out_of_range_rejection();
    test_late_counter_split();
    test_no_csv_path();
    test_default_unbounded_window();
    test_explicit_window_fail_closed();
    test_queue_index_lifetime_and_fail_closed();
    test_out_of_order_consumption_bound();
    test_calendar_out_of_order_turn0();
    test_first_block_over_window();
    test_arrival_zero_boundary();
    test_same_tick_multi_session();
    test_provenance_tampering();
    test_structure_validation_fail_closed();
    test_gate_trips_on_delayed_submission();
    test_future_arrival_clamp_and_consumption_leg();

    if (!g_ok) {
        std::fprintf(stderr, "[windowed_reader_test] FAIL: see messages "
                             "above\n");
        return 1;
    }
    std::printf("[windowed_reader_test] ALL PASS: calendar submission / row "
                "semantics / out-of-range rejection / split late counters / "
                "no-CSV path / default unbounded window / explicit-window "
                "fail-closed audit / queue-index pre-registration / "
                "out-of-order outstanding / out-of-order turn-0 calendar / "
                "long first block / t0 boundary / same-tick ordering / "
                "provenance tampering / structure validation / arrival "
                "gate self-test / future-alarm clamp + consumption leg\n");
    return 0;
}
