/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

windowed_trace_reader_test.cc -- phase-7 §10.4 WindowedTraceReader fixture.

Exercises the bounded-window CSV reader (方案 §10.4) standalone -- no
network simulation, no baseline artifacts touched:

  Part A  Window topping: the reader reads at most high_water un-consumed
          rows per pump; a full window does not read; consuming rows
          (notify_consumed) lets the next pump top up; the peak occupancy
          never exceeds high_water.
  Part B  Row semantics: turn-0 rows are submitted exactly once (command
          queue count == turn-0 row count), turn>0 rows are registered but
          never submitted (future_alarm path), EOF yields data_rows == total
          data rows and pump() == false.
  Part C  Out-of-range rejection: a turn-0 arrival beyond max_arrival_ns is
          rejected (never submitted, counted); in-range rows still submit.
  Part G  Backport fix 2026-08-16 (对比报告 §5.1) fail-closed window
          semantics, synthetic fixture (no 4-tier csv needed; the synthetic
          envelopes may use ANY arrival window): (i) with the DEFAULT
          arrival bound (now 0 = unbounded) every turn-0 row is accepted,
          however far out its arrival is; (ii) with an EXPLICIT small window
          the out-of-range rows are rejected AND counted, the rejected rows
          are consumed at reject time (the window does not clog: EOF is
          still reached, so the run-end completion audit denominator stays
          the materialized input's total row count and main_online's
          input-window audit fails the run on the nonzero rejection).
  Part D  Late-arrival clamp counter (RequestIngress): an arrival alarm whose
          target tick is already past is clamped to current+1 and counted.
  Part E  Request-neutral no-CSV path: an empty path constructs an
          already-EOF reader (no-op pumps, zero rows).
  Part F  Phase-7 §10.5 window-position checkpoint: write -> restore
          roundtrip (bookkeeping identical, restored reader keeps pumping
          from its open file position); missing file -> false; corrupt file
          -> false with state untouched (fail-closed); configuration
          mismatch (different high_water) -> false.

Build: the CMake target AstraSim_Analytical_Congestion_Aware_WindowedReaderTest.
Run (from template/astra-sim-sh_3.0):
  build/astra_analytical/build_congestion_aware/bin/\
    AstraSim_Analytical_Congestion_Aware_WindowedReaderTest
*******************************************************************************/

#include <cassert>
#include <cstdio>
#include <unistd.h>
#include <string>
#include <fstream>
#include <string>
#include <vector>

#include "astra-sim/workload/execution_driven/DecisionMailbox.hh"
#include "astra-sim/workload/execution_driven/RequestIngress.hh"
#include "astra-sim/workload/execution_driven/ServiceCoordinator.hh"
#include "astra-sim/workload/execution_driven/WindowedTraceReader.hh"
#include "common/EventQueue.h"

using namespace AstraSim::ExecutionDriven;
using namespace NetworkAnalytical;

namespace {

static std::string fixture_path(const char* base, const char* ext = ".csv") {
    // Parallel-reader validation (phase 5-6): per-process fixture paths so
    // concurrent instances do not race on shared /tmp files (fixture-only).
    return std::string("/tmp/") + base + "_" + std::to_string(::getpid()) + ext;
}


bool g_ok = true;

void expect(const bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "[windowed_reader_test] FAIL: %s\n", what);
        g_ok = false;
    }
}

void write_csv(const std::string& path,
               const std::vector<std::string>& rows) {
    std::ofstream out(path);
    out << "session_id,turn_index,request_id,prefill_length,decode_length,"
           "session_arrival_time_ns,inter_request_interval_ns,description\n";
    for (const auto& row : rows) {
        out << row << "\n";
    }
}

// 10 turn-0 rows (arrival 1..10 s, monotonic) + 1 turn>0 row (empty arrival).
std::vector<std::string> sample_rows() {
    std::vector<std::string> rows;
    for (int i = 0; i < 10; ++i) {
        rows.push_back("session_" + std::to_string(i) + ",0,session_" +
                       std::to_string(i) + "_request_0,100,50," +
                       std::to_string(1000000000ULL * (i + 1)) + ",,d");
    }
    rows.push_back("session_0,1,session_0_request_1,100,50,,"
                   "5000000000,d");
    return rows;
}

void test_window_topping() {
    const std::string csv = fixture_path("windowed_reader_test_window");
    write_csv(csv, sample_rows());

    EventQueue eq;
    DecisionMailbox mailbox;
    ServiceCoordinator svc;
    RequestIngress ingress;
    ingress.bind(&eq, &mailbox, &svc);

    WindowedTraceReader reader(csv, ingress, /*high_water=*/4);
    // Initial pump: reads up to 4 rows (4 turn-0 Submits queued).
    expect(reader.pump(), "initial pump still has rows");
    expect(reader.rows_read() == 4, "initial pump reads exactly 4 rows");
    expect(reader.high_water() == 4, "high_water reported");
    expect(ingress.pending_command_count() == 4,
           "4 turn-0 Submits queued by initial pump");
    expect(reader.peak_window_occupancy() <= 4,
           "peak occupancy within high_water");

    // Full window: another pump must not read anything.
    const size_t pumps_after_full = reader.read_pumps();
    expect(reader.pump(), "pump with full window still has rows");
    expect(reader.rows_read() == 4, "full window does not read");
    expect(reader.read_pumps() == pumps_after_full,
           "no read pump when the window is full");

    // Consume two rows -> occupancy 2 -> pump tops up to 4 (reads 2).
    reader.notify_consumed(0);
    reader.notify_consumed(1);
    reader.pump();
    expect(reader.rows_read() == 6, "consuming rows lets the window top up");

    // Drain everything: consume all rows read so far and pump until EOF.
    for (int64_t i = 0; i < static_cast<int64_t>(reader.rows_read()); ++i) {
        reader.notify_consumed(i);
    }
    while (reader.pump()) {
        for (int64_t i = 0; i < static_cast<int64_t>(reader.rows_read());
             ++i) {
            reader.notify_consumed(i);
        }
    }
    expect(reader.eof(), "EOF reached after consuming all rows");
    expect(reader.data_rows() == 11, "data_rows == total data rows at EOF");
    expect(reader.rows_read() == 11, "rows_read == total data rows at EOF");
    expect(reader.peak_window_occupancy() <= 4,
           "peak occupancy never exceeds high_water");
    expect(reader.rejected_out_of_range() == 0, "no rejections in part A");
    std::printf("[fixture] part A PASS: window topping / consumption\n");
}

void test_row_semantics() {
    const std::string csv = fixture_path("windowed_reader_test_semantics");
    write_csv(csv, sample_rows());

    EventQueue eq;
    DecisionMailbox mailbox;
    ServiceCoordinator svc;
    RequestIngress ingress;
    ingress.bind(&eq, &mailbox, &svc);

    // Unbounded window (0): one pump reads the whole file -- the full-pass
    // control arm of the phase-7 §10.4 window benchmark.
    WindowedTraceReader reader(csv, ingress, /*high_water=*/0);
    expect(!reader.pump(), "unbounded window EOFs in one pump");
    expect(reader.eof(), "unbounded window reaches EOF");
    expect(reader.data_rows() == 11, "data_rows == 11");
    expect(reader.rows_read() == 11, "rows_read == 11");
    expect(ingress.pending_command_count() == 10,
           "exactly the 10 turn-0 rows are submitted; turn>0 never is");
    std::printf("[fixture] part B PASS: turn-0 Submit / turn>0 future-alarm "
                "semantics\n");
}

void test_out_of_range_rejection() {
    const std::string csv = fixture_path("windowed_reader_test_reject");
    write_csv(csv, {
        "session_0,0,session_0_request_0,100,50,20000000000,,d",
        "session_1,0,session_1_request_0,100,50,40000000000,,d",
    });

    EventQueue eq;
    DecisionMailbox mailbox;
    ServiceCoordinator svc;
    RequestIngress ingress;
    ingress.bind(&eq, &mailbox, &svc);

    // max_arrival_ns = 30s: row 1 (20s) in range, row 2 (40s) rejected.
    WindowedTraceReader reader(csv, ingress, /*high_water=*/0,
                               /*max_arrival_ns=*/30000000000ULL);
    reader.pump();
    expect(reader.rejected_out_of_range() == 1,
           "out-of-range turn-0 row is rejected and counted");
    expect(reader.rows_read() == 2, "rejected row still counted as read");
    expect(ingress.pending_command_count() == 1,
           "only the in-range row is submitted");
    std::printf("[fixture] part C PASS: simulation-out-of-range rejection\n");
}

void test_late_arrival_clamp_counter() {
    EventQueue eq;
    DecisionMailbox mailbox;
    ServiceCoordinator svc;
    RequestIngress ingress;
    ingress.bind(&eq, &mailbox, &svc);

    // First: an in-time arrival (alarm at 1000 > current 0) is NOT clamped.
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

    // Advance the queue to t=1000 (the alarm fires; harmless without a
    // hook), then submit a request whose target tick (1000) is already
    // past: the drain clamps it to current+1 and counts one late arrival.
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
    expect(ingress.late_arrival_count() == 1,
           "late arrival clamp counted exactly once");

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
    std::printf("[fixture] part D PASS: late-arrival clamp counter\n");
}

void test_checkpoint_roundtrip() {
    const std::string csv = fixture_path("windowed_reader_test_checkpoint");
    write_csv(csv, sample_rows());
    const std::string cp = fixture_path("windowed_reader_test_checkpoint", ".json");

    // Build a reader, read 6 rows (consumed 0..3), checkpoint it.
    {
        EventQueue eq;
        DecisionMailbox mailbox;
        ServiceCoordinator svc;
        RequestIngress ingress;
        ingress.bind(&eq, &mailbox, &svc);
        WindowedTraceReader reader(csv, ingress, /*high_water=*/4);
        reader.pump();  // reads rows 0..3 (occupancy 4)
        for (int64_t i = 0; i < 4; ++i) {
            reader.notify_consumed(i);  // occupancy 0
        }
        reader.pump();  // tops up to rows 4..7 (occupancy 4)
        expect(reader.rows_read() == 8, "pre-checkpoint rows_read == 8");
        expect(reader.write_checkpoint(cp), "checkpoint written");
    }

    // Restore into a fresh reader over the same CSV: bookkeeping identical
    // (rows_read/data_rows/consumed), and pumping continues from the open
    // file position (occupancy accounting stays correct; no re-read of the
    // consumed prefix).
    {
        EventQueue eq;
        DecisionMailbox mailbox;
        ServiceCoordinator svc;
        RequestIngress ingress;
        ingress.bind(&eq, &mailbox, &svc);
        WindowedTraceReader reader(csv, ingress, /*high_water=*/4);
        expect(reader.read_checkpoint(cp), "checkpoint restored");
        expect(reader.rows_read() == 8, "rows_read restored");
        expect(reader.data_rows() == 8, "data_rows restored");
        expect(reader.high_water() == 4, "high_water unchanged by restore");
        // The restored reader's occupancy is 4 (8 read, 4 consumed) = full;
        // consume rows 4..7 and pump again: rows 8..11 arrive without any
        // re-read of the consumed prefix 0..7.
        for (int64_t i = 4; i < 8; ++i) {
            reader.notify_consumed(i);
        }
        expect(reader.pump(), "restored reader keeps pumping");
        expect(reader.rows_read() == 12, "pump continues from open position");
        expect(reader.peak_window_occupancy() <= 4,
               "peak occupancy still bounded after restore");
        std::remove(cp.c_str());
    }

    // Missing file -> false (nothing to restore).
    {
        EventQueue eq;
        DecisionMailbox mailbox;
        ServiceCoordinator svc;
        RequestIngress ingress;
        ingress.bind(&eq, &mailbox, &svc);
        WindowedTraceReader reader(csv, ingress, /*high_water=*/4);
        expect(!reader.read_checkpoint("/tmp/no_such_checkpoint.json"),
               "missing checkpoint returns false");
        expect(reader.rows_read() == 0, "missing checkpoint leaves state");
    }

    // Corrupt file -> false, state untouched (fail-closed).
    {
        EventQueue eq;
        DecisionMailbox mailbox;
        ServiceCoordinator svc;
        RequestIngress ingress;
        ingress.bind(&eq, &mailbox, &svc);
        WindowedTraceReader reader(csv, ingress, /*high_water=*/4);
        {
            std::ofstream bad(fixture_path("windowed_reader_test_bad_cp").c_str());
            bad << "{ not json\n";
        }
        expect(!reader.read_checkpoint(
                   fixture_path("windowed_reader_test_bad_cp").c_str()),
               "corrupt checkpoint returns false");
        expect(reader.rows_read() == 0, "corrupt checkpoint leaves state");
        std::remove(fixture_path("windowed_reader_test_bad_cp").c_str());
    }

    // Configuration mismatch (different high_water) -> false, state
    // untouched.
    {
        EventQueue eq;
        DecisionMailbox mailbox;
        ServiceCoordinator svc;
        RequestIngress ingress;
        ingress.bind(&eq, &mailbox, &svc);
        // Checkpoint above was written with high_water 4 and consumed; a
        // reader configured with high_water 0 must reject it.
        {
            std::ofstream cp_out(cp);
            cp_out << "{\"schema\":1,\"kind\":\"windowed_reader_checkpoint\","
                      "\"high_water\":4,\"max_arrival_ns\":30000000000,"
                      "\"header_seen\":true,\"eof\":false,\"data_rows\":6,"
                      "\"consumed_idx\":3,\"rows_read\":6,"
                      "\"rejected_out_of_range\":0,\"read_pumps\":2,"
                      "\"peak_occupancy\":4}\n";
        }
        WindowedTraceReader reader(csv, ingress, /*high_water=*/0);
        expect(!reader.read_checkpoint(cp),
               "config-mismatch checkpoint rejected (fail-closed)");
        expect(reader.rows_read() == 0,
               "config-mismatch checkpoint leaves state");
        std::remove(cp.c_str());
    }
    std::printf("[fixture] part F PASS: checkpoint roundtrip / missing / "
                "corrupt / config-mismatch fail-closed\n");
}

void test_no_csv_path() {
    EventQueue eq;
    DecisionMailbox mailbox;
    ServiceCoordinator svc;
    RequestIngress ingress;
    ingress.bind(&eq, &mailbox, &svc);

    WindowedTraceReader reader("", ingress, /*high_water=*/128);
    expect(reader.eof(), "empty path constructs an already-EOF reader");
    expect(!reader.pump(), "pump on the no-CSV reader returns false");
    expect(reader.data_rows() == 0, "no-CSV reader reads zero rows");
    expect(ingress.pending_command_count() == 0,
           "no-CSV reader submits nothing");
    std::printf("[fixture] part E PASS: request-neutral no-CSV path\n");
}

// Part G (backport fix 2026-08-16, 对比报告 §5.1): fail-closed window
// semantics on a synthetic queue -- arrivals far beyond any acceptance
// window (40 s / 60 s / 90 s) so the fixture does not depend on the 4-tier
// csv inputs.
void test_backport_window_fail_closed() {
    const std::string csv = fixture_path("windowed_reader_test_backport");
    write_csv(csv, {
        "session_0,0,session_0_request_0,100,50,40000000000,,d",
        "session_0,1,session_0_request_1,100,50,,5000000000,d",
        "session_1,0,session_1_request_0,100,50,60000000000,,d",
        "session_2,0,session_2_request_0,100,50,90000000000,,d",
    });

    // (i) DEFAULT arrival bound (0 = unbounded): every turn-0 row accepted,
    // however far out its arrival; zero rejections.
    {
        EventQueue eq;
        DecisionMailbox mailbox;
        ServiceCoordinator svc;
        RequestIngress ingress;
        ingress.bind(&eq, &mailbox, &svc);
        WindowedTraceReader reader(csv, ingress, /*high_water=*/2);
        // Default-constructed bound: no max_arrival_ns argument. Bounded
        // window exercises the pump/consume path while all 3 turn-0 rows
        // (40 s / 60 s / 90 s) sit far beyond the old 30e9 default.
        while (reader.pump()) {
            for (int64_t i = 0;
                 i < static_cast<int64_t>(reader.rows_read()); ++i) {
                reader.notify_consumed(i);
            }
        }
        expect(reader.eof(), "default unbounded: EOF reached");
        expect(reader.rejected_out_of_range() == 0,
               "default unbounded: zero rejections (no silent window)");
        expect(reader.data_rows() == 4, "default unbounded: all rows read");
        expect(ingress.pending_command_count() == 3,
               "default unbounded: every turn-0 row submitted");
    }

    // (ii) EXPLICIT small window (30 s): 3 turn-0 rows are out of range.
    {
        EventQueue eq;
        DecisionMailbox mailbox;
        ServiceCoordinator svc;
        RequestIngress ingress;
        ingress.bind(&eq, &mailbox, &svc);
        WindowedTraceReader reader(csv, ingress, /*high_water=*/2,
                                   /*max_arrival_ns=*/30000000000ULL);
        // One pump, no alarm ever fired: the rejected rows are consumed at
        // reject time, so even with high_water=2 the reader reads straight
        // through to EOF (the pre-backport clog stopped the window at 2
        // un-consumed rows and EOF was NEVER reached -- expected_requests
        // stayed 0 and the run-end audit was silently skipped).
        reader.pump();
        expect(reader.eof(),
               "explicit window: EOF still reached (rejected rows consumed "
               "at reject time; window never clogs)");
        expect(reader.rejected_out_of_range() == 3,
               "explicit window: all 3 out-of-range turn-0 rows counted");
        expect(reader.data_rows() == 4,
               "audit denominator: data_rows == materialized total rows "
               "(rejected rows included)");
        expect(reader.rows_read() == 4,
               "audit denominator: rows_read == materialized total rows");
        expect(ingress.pending_command_count() == 0,
               "explicit window: no turn-0 row in range -> nothing "
               "submitted (the nonzero rejection + un-drained run state is "
               "what main_online's input-window audit fails the run on)");
    }
    std::printf("[fixture] part G PASS: backport fail-closed window "
                "(default unbounded accepts all; explicit window counts "
                "drops, EOF reachable, denominator = total rows)\n");
}

}  // namespace

int main(int /*argc*/, char* /*argv*/[]) {
    test_window_topping();
    test_row_semantics();
    test_out_of_range_rejection();
    test_late_arrival_clamp_counter();
    test_checkpoint_roundtrip();
    test_no_csv_path();
    test_backport_window_fail_closed();

    if (!g_ok) {
        std::fprintf(stderr, "[windowed_reader_test] FAIL: see messages "
                             "above\n");
        return 1;
    }
    std::printf("[windowed_reader_test] ALL PASS: window topping / row "
                "semantics / out-of-range rejection / late clamp counter / "
                "checkpoint roundtrip / no-CSV path / backport fail-closed "
                "window\n");
    return 0;
}
