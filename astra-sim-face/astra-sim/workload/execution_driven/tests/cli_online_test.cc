/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

cli_online_test.cc -- phase-1 step 1-2 online CLI contract unit test.

Rules under test (方案 §4 步骤 1-2 操作 5, amended step 1-8 for the mode
token):
  R1  --online-mode is required for the online entry (missing = error).
  R2  --online-mode takes the mode token "strategy" (both the separate-value
      and inline "=" forms are legal; a missing value or any other token --
      including "replay", removed with the replay route on 2026-08-18 -- is a
      hard error).
  R3  request-neutral default: without --request-queue-csv the service stays
       IDLE (option simply absent; nothing is loaded).
  R4  --request-queue-csv must name an existing, readable file.
  R5  --close-input is a boolean flag (value form = error).
  R6  --bridge-dir is parsed and stored (reserved for step 1-7).
  R7  unknown options in the online family (--request-* / --bridge-* /
       --close-* / --online-* / --command-*) are hard errors (the shared
       CmdLineParser would silently swallow them).
  R8  a missing value after --bridge-dir / --request-queue-csv /
       --command-fifo = error.
  R9  step 1-10: --command-fifo is parsed and stored (external-producer
       FIFO for the IDLE fixture; both inline and separate-value forms).
  R10 phase 3: --sensing-enabled is a boolean feature flag, default OFF
       (感知开关必须经显式 feature flag 进入,阶段 6 前默认关); value form
       rejected; typos in the --sensing- family are hard errors.
  R11 wedge watchdog (2026-08-22 wedge-diagnosis recommendation):
       --bridge-timeout-ms is a non-negative integer (0 = wait forever,
       the frozen default; both inline and separate value forms legal;
       negative / garbage / over-int-range values are hard errors).
  R12 B.3 cleanup (2026-09-05): the --online-node-gc arm was removed (M2
       node GC is always on, amortized). The literal flag -- both value
       forms -- is an unknown online-family option now: stale scripts
       fail closed at parse time instead of silently disabling collection.
  R13 FP1 hardened unsigned lexicon (2026-09-01, sync-A16 batch P; E25/E32):
       every integer option (--request-max-arrival-ns
       / --bridge-timeout-ms) rejects leading-whitespace
       negatives (" -1", "\t-1"), explicit "+1", ERANGE-saturating tokens
       (ULLONG_MAX+1, 40-digit strings) BEFORE any value is written; the
       int-typed option rejects INT_MAX+1 and accepts exactly INT_MAX; any
       failed parse leaves the out struct untouched (sentinel check).
       (B.2 cleanup, 2026-09-05: --online-validate left the integer-lexicon
       family -- it is now a strict <0|1> enum, see R13b.)
  R13b B.2 cleanup (2026-09-05): --online-validate is a strict <0|1> enum
       (the N >= 2 sample-every-Nth-batch tier is removed). 0/1 accept in
       both value forms; every other token -- including the former sampling
       N >= 2, the wrap-around 4294967296, and whitespace/sign shapes -- is
       a hard parse error with no partial write into `out`.
  R14 FP1 watchdog lexicon and bounds (E26 ordering): --idle-watchdog-s
       accepts 0 (off), sub-second values, 1e-9 and 1e9 (the bounds
       themselves); rejects tokens containing whitespace or a sign
       character (" +1", " -1", "\t-1"), nan/inf/garbage tails, ERANGE
       under/overflow, (0, 1e-9) "below clock resolution", and anything
       above 1e9; failed parses leave out untouched.

Build: registered in the CMake build (M12②) as target
  AstraSim_Analytical_Congestion_Aware_CliOnlineTest in
  astra-sim/network_frontend/analytical/CMakeLists.txt (same shared-source +
  execution_driven recipe as the sibling fixtures; links AstraSim and
  Analytical_Congestion_Aware). Configure per README §2 (the
  build/astra_analytical aggregation with
  -DNETWORK_BACKEND_BUILD_AS_LIBRARY=ON), then:
    cmake --build build/astra_analytical/build_congestion_aware \
          --target AstraSim_Analytical_Congestion_Aware_CliOnlineTest -j
    build/astra_analytical/build_congestion_aware/bin/\
AstraSim_Analytical_Congestion_Aware_CliOnlineTest
  (the binary is emitted to <build-tree>/bin/ via the targets'
  RUNTIME_OUTPUT_DIRECTORY ../bin; run it with no arguments)

Assertions use the runtime expect() helper (same pattern as
windowed_trace_reader_test.cc) so they survive -DNDEBUG Release builds.
*******************************************************************************/

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "astra-sim/workload/execution_driven/OnlineCli.hh"

using namespace AstraSim::ExecutionDriven;

namespace {

bool g_ok = true;

void expect(const bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "[cli_online_test] FAIL: %s\n", what);
        g_ok = false;
    }
}

struct Args {
    std::vector<std::string> tokens;
    // argv[0] is a fake program name; existing/readable-file tokens
    // (/proc/self/exe) are passed explicitly by the caller.
    int argc;
    std::vector<char*> argv;

    explicit Args(std::vector<std::string> toks) : tokens(std::move(toks)) {
        tokens.insert(tokens.begin(), "fake_online_bin");
        argc = static_cast<int>(tokens.size());
        for (auto& t : tokens) {
            argv.push_back(const_cast<char*>(t.c_str()));
        }
    }
};

bool parse_ok(const std::vector<std::string>& toks, OnlineCliOptions& out) {
    Args a(toks);
    std::string error;
    return parse_online_cli(a.argc, a.argv.data(), out, error);
}

std::string parse_error(const std::vector<std::string>& toks) {
    Args a(toks);
    OnlineCliOptions out;
    std::string error;
    expect(!parse_online_cli(a.argc, a.argv.data(), out, error),
           "parse_error helper: the token list fails to parse");
    return error;
}

}  // namespace

int main() {
    OnlineCliOptions out;

    // R1: missing --online-mode is a hard error
    expect(parse_error({}).find("--online-mode") != std::string::npos,
           "R1: missing --online-mode is a hard error");

    // R2: mode token required
    expect(parse_error({"--online-mode"})
               .find("requires a value") != std::string::npos,
           "R2: bare --online-mode requires a value");
    // R2: valid token, separate-value form
    expect(parse_ok({"--online-mode", "strategy"}, out),
           "R2: valid token parses (separate-value form)");
    expect(out.mode == "strategy",
           "R2: separate-value form stores mode=strategy");
    // R2: valid token, inline "=" form
    expect(parse_ok({"--online-mode=strategy"}, out),
           "R2: valid token parses (inline \"=\" form)");
    expect(out.mode == "strategy", "R2: inline form stores mode=strategy");
    // R2: the replay token was removed with the replay route (2026-08-18)
    expect(parse_error({"--online-mode", "replay"})
               .find("unknown --online-mode value") != std::string::npos,
           "R2: replay token rejected (separate-value form)");
    expect(parse_error({"--online-mode=replay"})
               .find("unknown --online-mode value") != std::string::npos,
           "R2: replay token rejected (inline form)");
    // R2: any other token is a hard error
    expect(parse_error({"--online-mode", "bogus"})
               .find("unknown --online-mode value") != std::string::npos,
           "R2: bogus token rejected (separate-value form)");
    expect(parse_error({"--online-mode=offline"})
               .find("unknown --online-mode value") != std::string::npos,
           "R2: offline token rejected (inline form)");

    // R3: --online-mode strategy alone -> request-neutral defaults (IDLE; no
    // CSV, no command FIFO)
    expect(parse_ok({"--online-mode", "strategy"}, out),
           "R3: strategy alone parses (request-neutral defaults)");
    expect(out.mode == "strategy" && !out.close_input &&
               out.bridge_dir.empty() &&
               out.request_queue_csv.empty() && out.command_fifo.empty(),
           "R3: no CSV, no command FIFO, no bridge dir, close_input off");
    // 2026-09-05 default pin (R3): the parse path writes back its local
    // default unconditionally, so the BARE-invocation behavioral default of
    // the parking watchdog is armed at 1.0 s (explicit 0 = off, R14).
    expect(out.idle_watchdog_s == 1.0,
           "R3: bare invocation arms the parking watchdog at 1.0 s");

    // R4: existing readable CSV accepted
    expect(parse_ok({"--online-mode", "strategy",
                     "--request-queue-csv=/proc/self/exe"}, out),
           "R4: existing readable CSV accepted (inline form)");
    expect(out.request_queue_csv == "/proc/self/exe",
           "R4: inline form stores the CSV path");
    // R4: space-separated value form accepted
    expect(parse_ok({"--online-mode", "strategy", "--request-queue-csv",
                     "/proc/self/exe"}, out),
           "R4: existing readable CSV accepted (separate-value form)");
    expect(out.request_queue_csv == "/proc/self/exe",
           "R4: separate-value form stores the CSV path");
    // R4: missing file rejected
    expect(parse_error({"--online-mode", "strategy",
                        "--request-queue-csv=/no/such/file.csv"})
               .find("not readable") != std::string::npos,
           "R4: missing file rejected");

    // R5: --close-input flag
    expect(parse_ok({"--online-mode", "strategy", "--close-input"}, out),
           "R5: --close-input flag parses");
    expect(out.close_input, "R5: --close-input sets close_input");
    // R5: value form rejected
    expect(parse_error({"--online-mode", "strategy", "--close-input=true"})
               .find("no value") != std::string::npos,
           "R5: --close-input value form rejected");

    // R6: --bridge-dir parsed and stored
    expect(parse_ok({"--online-mode", "strategy", "--bridge-dir=/tmp/bridge"},
                    out),
           "R6: --bridge-dir accepted (inline form)");
    expect(out.bridge_dir == "/tmp/bridge",
           "R6: inline form stores bridge_dir");
    // R6: space-separated form
    expect(parse_ok({"--online-mode", "strategy", "--bridge-dir", "/tmp/bridge"},
                    out),
           "R6: --bridge-dir accepted (separate-value form)");
    expect(out.bridge_dir == "/tmp/bridge",
           "R6: separate-value form stores bridge_dir");

    // R11: --bridge-timeout-ms parsed and stored (both value forms); default
    // 0 = wait forever; negative / garbage / over-int-range are hard errors.
    expect(parse_ok({"--online-mode", "strategy",
                     "--bridge-timeout-ms=300000"}, out),
           "R11: --bridge-timeout-ms=300000 accepted (inline form)");
    expect(out.bridge_timeout_ms == 300000,
           "R11: inline form stores bridge_timeout_ms=300000");
    expect(parse_ok({"--online-mode", "strategy", "--bridge-timeout-ms",
                     "300000"}, out),
           "R11: --bridge-timeout-ms 300000 accepted (separate-value form)");
    expect(out.bridge_timeout_ms == 300000,
           "R11: separate-value form stores bridge_timeout_ms=300000");
    expect(parse_ok({"--online-mode", "strategy"}, out),
           "R11: default parse succeeds");
    expect(out.bridge_timeout_ms == 0,
           "R11: default bridge_timeout_ms is 0 (wait forever)");
    expect(parse_error({"--online-mode", "strategy",
                        "--bridge-timeout-ms=-1"})
               .find("non-negative integer") != std::string::npos,
           "R11: negative --bridge-timeout-ms rejected");
    expect(parse_error({"--online-mode", "strategy", "--bridge-timeout-ms=x"})
               .find("non-negative integer") != std::string::npos,
           "R11: garbage --bridge-timeout-ms rejected");
    expect(parse_error({"--online-mode", "strategy",
                        "--bridge-timeout-ms=99999999999"})
               .find("int range") != std::string::npos,
           "R11: over-int-range --bridge-timeout-ms rejected");

    // R7: typos in the online family are hard errors
    expect(parse_error({"--online-mode", "strategy", "--request-qeue-csv=/x"})
               .find("unknown online-family") != std::string::npos,
           "R7: typo --request-qeue-csv is a hard error");
    expect(parse_error({"--online-mode", "strategy", "--online-flag"})
               .find("unknown online-family") != std::string::npos,
           "R7: unknown --online-flag is a hard error");
    expect(parse_error({"--online-mode", "strategy", "--bridge-dirx=/x"})
               .find("unknown online-family") != std::string::npos,
           "R7: typo --bridge-dirx is a hard error");

    // R7: unrelated (non-online-family) options are left to the shared
    // CmdLineParser, not rejected here
    expect(parse_ok({"--online-mode", "strategy", "--comm-scale=2.0"}, out),
           "R7: non-online-family --comm-scale is left to the shared parser");

    // R8: missing value after --bridge-dir / --request-queue-csv /
    // --command-fifo
    expect(parse_error({"--online-mode", "strategy", "--bridge-dir"})
               .find("requires a value") != std::string::npos,
           "R8: bare --bridge-dir requires a value");
    expect(parse_error({"--online-mode", "strategy", "--request-queue-csv"})
               .find("requires a value") != std::string::npos,
           "R8: bare --request-queue-csv requires a value");
    expect(parse_error({"--online-mode", "strategy", "--command-fifo"})
               .find("requires a value") != std::string::npos,
           "R8: bare --command-fifo requires a value");

    // R9: --command-fifo parsed and stored (inline and separate-value forms)
    expect(parse_ok({"--online-mode", "strategy",
                     "--command-fifo=/tmp/fixture_cmd.fifo"}, out),
           "R9: --command-fifo accepted (inline form)");
    expect(out.command_fifo == "/tmp/fixture_cmd.fifo",
           "R9: inline form stores command_fifo");
    expect(parse_ok({"--online-mode", "strategy", "--command-fifo",
                     "/tmp/fixture_cmd.fifo"}, out),
           "R9: --command-fifo accepted (separate-value form)");
    expect(out.command_fifo == "/tmp/fixture_cmd.fifo",
           "R9: separate-value form stores command_fifo");
    // R7: --command-* typos are hard errors
    expect(parse_error({"--online-mode", "strategy", "--command-fifoX=/x"})
               .find("unknown online-family") != std::string::npos,
           "R7: typo --command-fifoX is a hard error");

    // missing --online-mode even with other online flags
    expect(parse_error({"--request-queue-csv=/proc/self/exe", "--close-input"})
               .find("--online-mode") != std::string::npos,
           "R1: missing --online-mode even with other online flags");

    // R11: phase-7 §10.4 --request-max-arrival-ns
    // (default: 0 = UNBOUNDED arrival window -- backport fix
    // 2026-08-16, sh_2.0测试 §5.1: the old 30e9 default burned the 30s
    // acceptance window into the code and silently dropped over-window
    // requests; the bound is now explicit-only and its drops fail-close
    // the run-end completion audit)
    expect(parse_ok({"--online-mode", "strategy"}, out),
           "R11: default parse succeeds (request-max-arrival-ns)");
    expect(out.request_max_arrival_ns == 0,
           "R11: default request_max_arrival_ns is 0 (UNBOUNDED)");
    expect(parse_ok({"--online-mode", "strategy",
                     "--request-max-arrival-ns=12345"}, out),
           "R11: --request-max-arrival-ns=12345 accepted");
    expect(out.request_max_arrival_ns == 12345,
           "R11: request_max_arrival_ns stores 12345");
    // R11: non-integer / negative values are hard errors
    expect(parse_error({"--online-mode", "strategy",
                        "--request-max-arrival-ns=-5"})
               .find("non-negative integer") != std::string::npos,
           "R11: negative --request-max-arrival-ns rejected");
    // R11: missing value forms rejected
    expect(parse_error({"--online-mode", "strategy",
                        "--request-max-arrival-ns"})
               .find("requires a value") != std::string::npos,
           "R11: bare --request-max-arrival-ns requires a value");
    // R11: typos in the new family are hard errors
    expect(parse_error({"--online-mode", "strategy",
                        "--request-window-row=64"})
               .find("unknown online-family") != std::string::npos,
           "R11: typo --request-window-row is a hard error");

    // R10: --sensing-enabled default OFF (feature flag; phase 6 before 默认关)
    expect(parse_ok({"--online-mode", "strategy"}, out),
           "R10: default parse succeeds (sensing, first)");
    expect(!out.sensing_enabled, "R10: --sensing-enabled default OFF");
    expect(parse_ok({"--online-mode", "strategy"}, out),
           "R10: default parse succeeds (sensing, repeat)");
    expect(!out.sensing_enabled, "R10: --sensing-enabled stays OFF");
    // R10: flag form enables
    expect(parse_ok({"--online-mode", "strategy", "--sensing-enabled"}, out),
           "R10: --sensing-enabled flag form parses");
    expect(out.sensing_enabled, "R10: flag form enables sensing");
    // R10: value form rejected
    expect(parse_error({"--online-mode", "strategy", "--sensing-enabled=true"})
               .find("takes no value") != std::string::npos,
           "R10: --sensing-enabled value form rejected");
    // R10: --sensing-* typos are hard errors
    expect(parse_error({"--online-mode", "strategy", "--sensing-enable=/x"})
               .find("unknown online-family") != std::string::npos,
           "R10: typo --sensing-enable is a hard error");
    expect(parse_error({"--online-mode", "strategy", "--sensing-x"})
               .find("unknown online-family") != std::string::npos,
           "R10: unknown --sensing-x is a hard error");
    // R10: coexists with the rest of the family
    expect(parse_ok({"--online-mode", "strategy", "--close-input",
                     "--sensing-enabled", "--bridge-dir=/tmp/bridge"}, out),
           "R10: --sensing-enabled coexists with the rest of the family");
    expect(out.sensing_enabled && out.close_input &&
           out.bridge_dir == "/tmp/bridge",
           "R10: family options all stored together");

    // R12 B.3 cleanup (2026-09-05): the --online-node-gc arm was removed;
    // M2 node GC is always on. The literal flag (both value forms) is an
    // unknown online-family option -- stale scripts fail closed at parse
    // time instead of silently disabling collection.
    expect(parse_error({"--online-mode", "strategy", "--online-node-gc=0"})
               .find("unknown online-family") != std::string::npos,
           "R12: removed --online-node-gc value form is a hard error");
    expect(parse_error({"--online-mode", "strategy", "--online-node-gc", "1"})
               .find("unknown online-family") != std::string::npos,
           "R12: removed --online-node-gc flag form is a hard error");

    // R13: FP1 hardened unsigned lexicon (2026-09-01, sync-A16 batch P;
    // E25/E32). The whitespace-negative shapes defeated the old
    // first-character '-' check (strtoull skips leading whitespace, parses
    // the negation, wraps to ULLONG_MAX and sets no ERANGE on 64-bit);
    // "+1" was silently accepted. All integer options now share the
    // pure-ASCII-digit lexicon and must reject every one of these shapes
    // with no partial write into `out`.
    const std::vector<std::string> int_opts = {
        "--request-max-arrival-ns", "--bridge-timeout-ms"};
    for (const std::string& opt : int_opts) {
        // sentinel: prove a failed parse writes nothing
        OnlineCliOptions sentinel;
        sentinel.request_max_arrival_ns = 777;
        sentinel.bridge_timeout_ms = 777;
        for (const std::string& bad : {" -1", "\t-1", "+1", "-1", "1x", " 1",
                                       "18446744073709551616",
                                       "9999999999999999999999999999999999"
                                       "999999999"}) {
            OnlineCliOptions probe = sentinel;
            expect(parse_error({"--online-mode", "strategy", opt, bad})
                       .find("integer") != std::string::npos,
                   "R13: bad integer token rejected with the lexicon error");
            // separate-value form reaches the same lexicon
            expect(!parse_ok({"--online-mode", "strategy", opt, bad}, probe),
                   "R13: bad integer token fails to parse (separate-value)");
            expect(probe.request_max_arrival_ns == 777 &&
                   probe.bridge_timeout_ms == 777,
                   "R13: failed parse leaves the sentinel untouched");
        }
    }
    // R13: int-typed options bound at exactly INT_MAX.
    expect(parse_ok({"--online-mode", "strategy",
                     "--bridge-timeout-ms=2147483647"}, out),
           "R13: --bridge-timeout-ms at INT_MAX accepted");
    expect(out.bridge_timeout_ms == 2147483647,
           "R13: bridge_timeout_ms stores INT_MAX exactly");
    expect(parse_error({"--online-mode", "strategy",
                        "--bridge-timeout-ms=2147483648"})
               .find("int range") != std::string::npos,
           "R13: INT_MAX+1 rejected (int range)");

    // R13b (B.2 cleanup, 2026-09-05): --online-validate is a strict <0|1>
    // enum; the N >= 2 sampled tier is removed. 0/1 accept in both value
    // forms; every other token (the sampling N, the wrap-around 4294967296,
    // and the sign/whitespace shapes the old numeric lexicon used to
    // reject) is a hard parse error with no partial write into `out`.
    {
        // sentinel: prove a failed parse writes nothing
        OnlineCliOptions sentinel;
        sentinel.online_validate = 777;
        expect(parse_ok({"--online-mode", "strategy", "--online-validate=0"},
                        out),
               "R13b: --online-validate=0 accepted");
        expect(out.online_validate == 0, "R13b: online_validate stores 0");
        expect(parse_ok({"--online-mode", "strategy", "--online-validate=1"},
                        out),
               "R13b: --online-validate=1 accepted");
        expect(out.online_validate == 1, "R13b: online_validate stores 1");
        OnlineCliOptions probe;
        expect(parse_ok({"--online-mode", "strategy", "--online-validate",
                         "0"},
                        probe),
               "R13b: separate-value 0 accepted");
        expect(probe.online_validate == 0,
               "R13b: separate-value form stores 0");
        expect(parse_ok({"--online-mode", "strategy", "--online-validate",
                         "1"},
                        probe),
               "R13b: separate-value 1 accepted");
        expect(probe.online_validate == 1,
               "R13b: separate-value form stores 1");
        for (const std::string& bad :
             {"2", "3", "01", "on", "true", "-1", "+1", "1x", " 1",
              "2147483647", "2147483648", "4294967296"}) {
            OnlineCliOptions p = sentinel;
            expect(parse_error({"--online-mode", "strategy",
                                "--online-validate=" + bad})
                       .find("unknown --online-validate value") !=
                   std::string::npos,
                   "R13b: bad --online-validate token rejected");
            // separate-value form reaches the same enum check
            expect(!parse_ok(
                {"--online-mode", "strategy", "--online-validate", bad}, p),
                   "R13b: bad token fails to parse (separate-value form)");
            expect(p.online_validate == 777,
                   "R13b: failed parse leaves the sentinel untouched");
        }
    }

    // R14: FP1 watchdog lexicon and E26 bounds ordering.
    // 2026-09-05 default pin: the struct default is armed at 1.0 s; the
    // explicit 0 = off escape is asserted immediately below.
    {
        OnlineCliOptions defaults_probe;
        expect(defaults_probe.idle_watchdog_s == 1.0,
               "R14: struct default idle_watchdog_s is 1.0 s");
    }
    // accepted: 0 = off, sub-second, both bounds themselves
    expect(parse_ok({"--online-mode", "strategy", "--idle-watchdog-s", "0"},
                    out),
           "R14: --idle-watchdog-s 0 (off) accepted");
    expect(out.idle_watchdog_s == 0.0, "R14: explicit 0 stores 0.0 (off)");
    expect(parse_ok({"--online-mode", "strategy", "--idle-watchdog-s=0.5"},
                    out),
           "R14: sub-second --idle-watchdog-s accepted");
    expect(out.idle_watchdog_s == 0.5,
           "R14: sub-second value stored exactly");
    {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.17g", kMinIdleWatchdogSeconds);
        expect(parse_ok({"--online-mode", "strategy", "--idle-watchdog-s",
                         buf}, out),
               "R14: lower bound kMinIdleWatchdogSeconds accepted");
        expect(out.idle_watchdog_s == kMinIdleWatchdogSeconds,
               "R14: lower bound stored exactly");
        std::snprintf(buf, sizeof(buf), "%.17g", kMaxIdleWatchdogSeconds);
        expect(parse_ok({"--online-mode", "strategy", "--idle-watchdog-s",
                         buf}, out),
               "R14: upper bound kMaxIdleWatchdogSeconds accepted");
        expect(out.idle_watchdog_s == kMaxIdleWatchdogSeconds,
               "R14: upper bound stored exactly");
    }
    // rejected lexicon: whitespace / sign characters anywhere
    for (const std::string& bad : {" +1", " -1", "\t-1", "+900", "-0.5",
                                   "1 ", "9 00"}) {
        OnlineCliOptions probe;
        probe.idle_watchdog_s = -42.5;
        expect(parse_error({"--online-mode", "strategy",
                            "--idle-watchdog-s", bad})
                   .find("whitespace or sign") != std::string::npos,
               "R14: whitespace/sign token rejected");
        expect(!parse_ok({"--online-mode", "strategy",
                          "--idle-watchdog-s", bad}, probe),
               "R14: whitespace/sign token fails to parse");
        expect(probe.idle_watchdog_s == -42.5,
               "R14: failed parse leaves probe untouched");
    }
    // rejected: garbage tails / non-finite / ERANGE both directions
    for (const std::string& bad : {"abc", "1e", "nan", "inf", "1e309",
                                   "1e-320", "0.0e0x"}) {
        expect(parse_error({"--online-mode", "strategy",
                            "--idle-watchdog-s", bad})
                   .find("finite number") != std::string::npos,
               "R14: garbage/non-finite/ERANGE token rejected");
    }
    // rejected between the bounds (E26 order): below clock resolution and
    // above the platform cap
    expect(parse_error({"--online-mode", "strategy",
                        "--idle-watchdog-s=1e-10"})
               .find("clock resolution") != std::string::npos,
           "R14: below clock resolution rejected (E26 order)");
    expect(parse_error({"--online-mode", "strategy",
                        "--idle-watchdog-s=9.3e9"})
               .find("upper bound") != std::string::npos,
           "R14: above the platform cap rejected (E26 order)");
    expect(parse_error({"--online-mode", "strategy",
                        "--idle-watchdog-s=1e308"})
               .find("upper bound") != std::string::npos,
           "R14: 1e308 rejected by the upper bound");
    // missing value
    expect(parse_error({"--online-mode", "strategy", "--idle-watchdog-s"})
               .find("requires a value") != std::string::npos,
           "R14: bare --idle-watchdog-s requires a value");
    // --idle-* typos stay hard errors (family fail-closed)
    expect(parse_error({"--online-mode", "strategy", "--idle-watchdog=1"})
               .find("unknown online-family") != std::string::npos,
           "R14: typo --idle-watchdog is a hard error");

    if (!g_ok) {
        std::fprintf(stderr,
                     "[cli_online_test] FAIL: see messages above\n");
        return 1;
    }
    std::printf("[cli] ALL PASS: R1-R14 online CLI contract verified "
                "(incl. phase-7 §10.4 window knobs, B.3 node-gc arm removal, FP1 "
                "hardened unsigned lexicon + watchdog bounds)\n");
    return 0;
}
