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
  R5  --close-input boolean semantics (P2-5 fix, 2026-09-28): unified
       forms -- bare flag = true, inline "=true|1"/"=false|0" and
       separated "true|1"/"false|0" parse explicitly with one meaning;
       any other value token is a hard error. (Pre-fix the inline form was
       rejected while a separated value was silently IGNORED, so
       "--close-input false" armed the close.)
  R6  --bridge-dir is parsed and stored (reserved for step 1-7).
  R7  unknown options in the online family (--request-* / --bridge-* /
       --close-* / --online-* / --command-*) are hard errors (the shared
       CmdLineParser fails closed on anything strip_online_family_argv
       leaves behind).
  R8  a missing value after --bridge-dir / --request-queue-csv /
       --command-fifo = error.
  R9  step 1-10: --command-fifo is parsed and stored (external-producer
       FIFO for the IDLE fixture; both inline and separate-value forms).
  R10 phase 3: --sensing-enabled is a boolean feature flag, default OFF
       (感知开关必须经显式 feature flag 进入,阶段 6 前默认关); P2-5 unified
       boolean semantics (same forms as R5); typos in the --sensing-
       family are hard errors.
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
  R15 P2-4 fix (2026-09-28): a value-taking online option never consumes a
       "--"-prefixed token as its value ("--bridge-dir --close-input" is a
       hard error, not a directory named "--close-input" plus a silently
       inverted close flag). R15b P2-6 fix: an explicitly given but EMPTY
       value ("--request-queue-csv=") is a hard error, distinct from the
       option being absent (the pre-fix shape skipped the readability
       check, ran expected_requests=0 and let every completion gate pass
       on a zero-request green run). R15c P2-4 path legality: --bridge-dir
       must be a directory when it exists (a missing path stays legal --
       ensure_bridge_dir creates it); --command-fifo must name an existing
       FIFO (the fixture mkfifs it before the binary starts).
  R16 C3 fix (2026-09-28): --close-input and --command-fifo are mutually
       exclusive (hard parse error): the startup close would kill the FIFO
       producer channel before it injects anything and the no-CSV run
       would exit green with zero requests. --close-input=false with a
       FIFO stays legal.

Build: registered in the CMake build (M20) as target
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
*******************************************************************************/

#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "astra-sim/workload/execution_driven/OnlineCli.hh"

using namespace AstraSim::ExecutionDriven;

namespace {

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
    assert(!parse_online_cli(a.argc, a.argv.data(), out, error));
    return error;
}

}  // namespace

int main() {
    OnlineCliOptions out;

    // R1: missing --online-mode is a hard error
    assert(parse_error({}).find("--online-mode") != std::string::npos);

    // R2: mode token required
    assert(parse_error({"--online-mode"})
               .find("requires a value") != std::string::npos);
    // R2: valid token, separate-value form
    assert(parse_ok({"--online-mode", "strategy"}, out));
    assert(out.mode == "strategy");
    // R2: valid token, inline "=" form
    assert(parse_ok({"--online-mode=strategy"}, out));
    assert(out.mode == "strategy");
    // R2: the replay token was removed with the replay route (2026-08-18)
    assert(parse_error({"--online-mode", "replay"})
               .find("unknown --online-mode value") != std::string::npos);
    assert(parse_error({"--online-mode=replay"})
               .find("unknown --online-mode value") != std::string::npos);
    // R2: any other token is a hard error
    assert(parse_error({"--online-mode", "bogus"})
               .find("unknown --online-mode value") != std::string::npos);
    assert(parse_error({"--online-mode=offline"})
               .find("unknown --online-mode value") != std::string::npos);

    // R3: --online-mode strategy alone -> request-neutral defaults (IDLE; no
    // CSV, no command FIFO)
    assert(parse_ok({"--online-mode", "strategy"}, out));
    assert(out.mode == "strategy" && !out.close_input &&
           out.bridge_dir.empty() &&
           out.request_queue_csv.empty() && out.command_fifo.empty());
    // 2026-09-05 default pin (R3): the parse path writes back its local
    // default unconditionally, so the BARE-invocation behavioral default of
    // the parking watchdog is armed at 1.0 s (explicit 0 = off, R14).
    assert(out.idle_watchdog_s == 1.0);

    // R4: existing readable CSV accepted
    assert(parse_ok({"--online-mode", "strategy",
                     "--request-queue-csv=/proc/self/exe"}, out));
    assert(out.request_queue_csv == "/proc/self/exe");
    // R4: space-separated value form accepted
    assert(parse_ok({"--online-mode", "strategy", "--request-queue-csv",
                     "/proc/self/exe"}, out));
    assert(out.request_queue_csv == "/proc/self/exe");
    // R4: missing file rejected
    assert(parse_error({"--online-mode", "strategy",
                        "--request-queue-csv=/no/such/file.csv"})
               .find("not readable") != std::string::npos);

    // R5 (P2-5, 2026-09-28): --close-input unified boolean semantics --
    // bare flag = true; inline "=true|1"/"=false|0"; separated
    // "true|1"/"false|0"; any other value token is a hard error.
    assert(parse_ok({"--online-mode", "strategy", "--close-input"}, out));
    assert(out.close_input);
    assert(parse_ok({"--online-mode", "strategy", "--close-input=true"},
                    out));
    assert(out.close_input);
    assert(parse_ok({"--online-mode", "strategy", "--close-input=1"}, out));
    assert(out.close_input);
    assert(parse_ok({"--online-mode", "strategy", "--close-input=false"},
                    out));
    assert(!out.close_input);
    assert(parse_ok({"--online-mode", "strategy", "--close-input=0"}, out));
    assert(!out.close_input);
    assert(parse_ok({"--online-mode", "strategy", "--close-input", "false"},
                    out));
    assert(!out.close_input);
    assert(parse_ok({"--online-mode", "strategy", "--close-input", "0"},
                    out));
    assert(!out.close_input);
    assert(parse_ok({"--online-mode", "strategy", "--close-input", "true"},
                    out));
    assert(out.close_input);
    assert(parse_ok({"--online-mode", "strategy", "--close-input", "1"},
                    out));
    assert(out.close_input);
    // a "--"-prefixed token after the bare flag is another option, not a
    // value: the flag stands alone (true)
    assert(parse_ok({"--online-mode", "strategy", "--close-input",
                     "--sensing-enabled"},
                    out));
    assert(out.close_input);
    // illegal values fail closed in both forms (pre-fix: the separated
    // shape was silently ignored and the flag armed)
    assert(parse_error({"--online-mode", "strategy", "--close-input=yes"})
               .find("requires a boolean value") != std::string::npos);
    assert(parse_error({"--online-mode", "strategy", "--close-input", "yes"})
               .find("requires a boolean value") != std::string::npos);
    {
        // the separated "false" form must explicitly WRITE false (the
        // pre-fix parser ignored the token and left the flag armed)
        OnlineCliOptions probe;
        probe.close_input = true;
        assert(parse_ok(
            {"--online-mode", "strategy", "--close-input", "false"}, probe));
        assert(!probe.close_input);
    }

    // R6: --bridge-dir parsed and stored
    assert(parse_ok({"--online-mode", "strategy", "--bridge-dir=/tmp/bridge"},
                    out));
    assert(out.bridge_dir == "/tmp/bridge");
    // R6: space-separated form
    assert(parse_ok({"--online-mode", "strategy", "--bridge-dir", "/tmp/bridge"},
                    out));
    assert(out.bridge_dir == "/tmp/bridge");

    // R11: --bridge-timeout-ms parsed and stored (both value forms); default
    // 0 = wait forever; negative / garbage / over-int-range are hard errors.
    assert(parse_ok({"--online-mode", "strategy",
                     "--bridge-timeout-ms=300000"}, out));
    assert(out.bridge_timeout_ms == 300000);
    assert(parse_ok({"--online-mode", "strategy", "--bridge-timeout-ms",
                     "300000"}, out));
    assert(out.bridge_timeout_ms == 300000);
    assert(parse_ok({"--online-mode", "strategy"}, out));
    assert(out.bridge_timeout_ms == 0);
    assert(parse_error({"--online-mode", "strategy",
                        "--bridge-timeout-ms=-1"})
               .find("non-negative integer") != std::string::npos);
    assert(parse_error({"--online-mode", "strategy", "--bridge-timeout-ms=x"})
               .find("non-negative integer") != std::string::npos);
    assert(parse_error({"--online-mode", "strategy",
                        "--bridge-timeout-ms=99999999999"})
               .find("int range") != std::string::npos);

    // R7: typos in the online family are hard errors
    assert(parse_error({"--online-mode", "strategy", "--request-qeue-csv=/x"})
               .find("unknown online-family") != std::string::npos);
    assert(parse_error({"--online-mode", "strategy", "--online-flag"})
               .find("unknown online-family") != std::string::npos);
    assert(parse_error({"--online-mode", "strategy", "--bridge-dirx=/x"})
               .find("unknown online-family") != std::string::npos);

    // R7: unrelated (non-online-family) options are left to the shared
    // CmdLineParser, not rejected here
    assert(parse_ok({"--online-mode", "strategy", "--comm-scale=2.0"}, out));

    // R8: missing value after --bridge-dir / --request-queue-csv /
    // --command-fifo
    assert(parse_error({"--online-mode", "strategy", "--bridge-dir"})
               .find("requires a value") != std::string::npos);
    assert(parse_error({"--online-mode", "strategy", "--request-queue-csv"})
               .find("requires a value") != std::string::npos);
    assert(parse_error({"--online-mode", "strategy", "--command-fifo"})
               .find("requires a value") != std::string::npos);

    // R9 (amended by R15c, P2-4): --command-fifo parsed and stored (inline
    // and separate-value forms). The value must name an EXISTING FIFO, so
    // the fixture mkfifs one (per-process path).
    const std::string fifo_path =
        "/tmp/cli_online_fixture_" + std::to_string(::getpid()) + ".fifo";
    // tests-F10 (2026-10-01): idempotent stale-FIFO guard -- pid reuse plus
    // a leftover FIFO from a previous assert-aborted run (the remove at the
    // end of main is skipped on abort) made mkfifo fail with EEXIST and
    // turned the assert into a false red.
    ::unlink(fifo_path.c_str());
    assert(::mkfifo(fifo_path.c_str(), 0600) == 0);
    assert(parse_ok({"--online-mode", "strategy",
                     "--command-fifo=" + fifo_path},
                    out));
    assert(out.command_fifo == fifo_path);
    assert(parse_ok({"--online-mode", "strategy", "--command-fifo",
                     fifo_path},
                    out));
    assert(out.command_fifo == fifo_path);
    // R15c: a missing path and an existing non-FIFO file are hard errors
    assert(parse_error({"--online-mode", "strategy",
                        "--command-fifo=/no/such/fixture_cmd.fifo"})
               .find("must name an existing FIFO") != std::string::npos);
    assert(parse_error({"--online-mode", "strategy",
                        "--command-fifo=/proc/self/exe"})
               .find("must name an existing FIFO") != std::string::npos);
    // R7: --command-* typos are hard errors
    assert(parse_error({"--online-mode", "strategy", "--command-fifoX=/x"})
               .find("unknown online-family") != std::string::npos);

    // R16 (C3 fix, 2026-09-28): --close-input and --command-fifo are
    // mutually exclusive -- the startup close would kill the FIFO producer
    // channel before it injects anything and the no-CSV run would exit
    // green with zero requests. Every boolean form that arms the close is
    // a hard parse error with the FIFO present; an explicit
    // --close-input=false with a FIFO stays legal.
    assert(parse_error({"--online-mode", "strategy",
                        "--command-fifo=" + fifo_path, "--close-input"})
               .find("mutually exclusive") != std::string::npos);
    assert(parse_error({"--online-mode", "strategy", "--command-fifo",
                        fifo_path, "--close-input=true"})
               .find("mutually exclusive") != std::string::npos);
    assert(parse_error({"--online-mode", "strategy",
                        "--command-fifo=" + fifo_path, "--close-input", "1"})
               .find("mutually exclusive") != std::string::npos);
    assert(parse_ok({"--online-mode", "strategy",
                     "--command-fifo=" + fifo_path, "--close-input=false"},
                    out));
    assert(out.command_fifo == fifo_path);
    assert(!out.close_input);

    // missing --online-mode even with other online flags
    assert(parse_error({"--request-queue-csv=/proc/self/exe", "--close-input"})
               .find("--online-mode") != std::string::npos);

    // R11: phase-7 §10.4 --request-max-arrival-ns
    // (default: 0 = UNBOUNDED arrival window -- backport fix
    // 2026-08-16, sh_2.0测试 §5.1: the old 30e9 default burned the 30s
    // acceptance window into the code and silently dropped over-window
    // requests; the bound is now explicit-only and its drops fail-close
    // the run-end completion audit)
    assert(parse_ok({"--online-mode", "strategy"}, out));
    assert(out.request_max_arrival_ns == 0);
    assert(parse_ok({"--online-mode", "strategy",
                     "--request-max-arrival-ns=12345"}, out));
    assert(out.request_max_arrival_ns == 12345);
    // R11: non-integer / negative values are hard errors
    assert(parse_error({"--online-mode", "strategy",
                        "--request-max-arrival-ns=-5"})
               .find("non-negative integer") != std::string::npos);
    // R11: missing value forms rejected
    assert(parse_error({"--online-mode", "strategy",
                        "--request-max-arrival-ns"})
               .find("requires a value") != std::string::npos);
    // R11: typos in the new family are hard errors
    assert(parse_error({"--online-mode", "strategy",
                        "--request-window-row=64"})
               .find("unknown online-family") != std::string::npos);

    // R10: --sensing-enabled default OFF (feature flag; phase 6 before 默认关)
    assert(parse_ok({"--online-mode", "strategy"}, out));
    assert(!out.sensing_enabled);
    assert(parse_ok({"--online-mode", "strategy"}, out));
    assert(!out.sensing_enabled);
    // R10: flag form enables
    assert(parse_ok({"--online-mode", "strategy", "--sensing-enabled"}, out));
    assert(out.sensing_enabled);
    // R10 (P2-5, 2026-09-28): unified boolean forms, same lexicon as R5
    assert(parse_ok({"--online-mode", "strategy", "--sensing-enabled=true"},
                    out));
    assert(out.sensing_enabled);
    assert(parse_ok({"--online-mode", "strategy", "--sensing-enabled=false"},
                    out));
    assert(!out.sensing_enabled);
    assert(parse_ok({"--online-mode", "strategy", "--sensing-enabled", "0"},
                    out));
    assert(!out.sensing_enabled);
    assert(parse_ok({"--online-mode", "strategy", "--sensing-enabled", "1"},
                    out));
    assert(out.sensing_enabled);
    // R10: illegal value tokens fail closed in both forms
    assert(parse_error({"--online-mode", "strategy",
                        "--sensing-enabled=on"})
               .find("requires a boolean value") != std::string::npos);
    assert(parse_error({"--online-mode", "strategy",
                        "--sensing-enabled", "on"})
               .find("requires a boolean value") != std::string::npos);
    // R10: --sensing-* typos are hard errors
    assert(parse_error({"--online-mode", "strategy", "--sensing-enable=/x"})
               .find("unknown online-family") != std::string::npos);
    assert(parse_error({"--online-mode", "strategy", "--sensing-x"})
               .find("unknown online-family") != std::string::npos);
    // R10: coexists with the rest of the family
    assert(parse_ok({"--online-mode", "strategy", "--close-input",
                     "--sensing-enabled", "--bridge-dir=/tmp/bridge"}, out));
    assert(out.sensing_enabled && out.close_input &&
           out.bridge_dir == "/tmp/bridge");

    // R12 B.3 cleanup (2026-09-05): the --online-node-gc arm was removed;
    // M2 node GC is always on. The literal flag (both value forms) is an
    // unknown online-family option -- stale scripts fail closed at parse
    // time instead of silently disabling collection.
    assert(parse_error({"--online-mode", "strategy", "--online-node-gc=0"})
               .find("unknown online-family") != std::string::npos);
    assert(parse_error({"--online-mode", "strategy", "--online-node-gc", "1"})
               .find("unknown online-family") != std::string::npos);

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
            assert(parse_error({"--online-mode", "strategy", opt, bad})
                       .find("integer") != std::string::npos);
            // separate-value form reaches the same lexicon
            assert(!parse_ok({"--online-mode", "strategy", opt, bad}, probe));
            assert(probe.request_max_arrival_ns == 777 &&
                   probe.bridge_timeout_ms == 777);
        }
    }
    // R13: int-typed options bound at exactly INT_MAX.
    assert(parse_ok({"--online-mode", "strategy",
                     "--bridge-timeout-ms=2147483647"}, out));
    assert(out.bridge_timeout_ms == 2147483647);
    assert(parse_error({"--online-mode", "strategy",
                        "--bridge-timeout-ms=2147483648"})
               .find("int range") != std::string::npos);

    // R13b (B.2 cleanup, 2026-09-05): --online-validate is a strict <0|1>
    // enum; the N >= 2 sampled tier is removed. 0/1 accept in both value
    // forms; every other token (the sampling N, the wrap-around 4294967296,
    // and the sign/whitespace shapes the old numeric lexicon used to
    // reject) is a hard parse error with no partial write into `out`.
    {
        // sentinel: prove a failed parse writes nothing
        OnlineCliOptions sentinel;
        sentinel.online_validate = 777;
        assert(parse_ok({"--online-mode", "strategy", "--online-validate=0"},
                        out));
        assert(out.online_validate == 0);
        assert(parse_ok({"--online-mode", "strategy", "--online-validate=1"},
                        out));
        assert(out.online_validate == 1);
        OnlineCliOptions probe;
        assert(parse_ok({"--online-mode", "strategy", "--online-validate",
                         "0"},
                        probe));
        assert(probe.online_validate == 0);
        assert(parse_ok({"--online-mode", "strategy", "--online-validate",
                         "1"},
                        probe));
        assert(probe.online_validate == 1);
        for (const std::string& bad :
             {"2", "3", "01", "on", "true", "-1", "+1", "1x", " 1",
              "2147483647", "2147483648", "4294967296"}) {
            OnlineCliOptions p = sentinel;
            assert(parse_error({"--online-mode", "strategy",
                                "--online-validate=" + bad})
                       .find("unknown --online-validate value") !=
                   std::string::npos);
            // separate-value form reaches the same enum check
            assert(!parse_ok(
                {"--online-mode", "strategy", "--online-validate", bad}, p));
            assert(p.online_validate == 777);
        }
    }

    // R14: FP1 watchdog lexicon and E26 bounds ordering.
    // 2026-09-05 default pin: the struct default is armed at 1.0 s; the
    // explicit 0 = off escape is asserted immediately below.
    {
        OnlineCliOptions defaults_probe;
        assert(defaults_probe.idle_watchdog_s == 1.0);
    }
    // accepted: 0 = off, sub-second, both bounds themselves
    assert(parse_ok({"--online-mode", "strategy", "--idle-watchdog-s", "0"},
                    out));
    assert(out.idle_watchdog_s == 0.0);
    assert(parse_ok({"--online-mode", "strategy", "--idle-watchdog-s=0.5"},
                    out));
    assert(out.idle_watchdog_s == 0.5);
    {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.17g", kMinIdleWatchdogSeconds);
        assert(parse_ok({"--online-mode", "strategy", "--idle-watchdog-s",
                         buf}, out));
        assert(out.idle_watchdog_s == kMinIdleWatchdogSeconds);
        std::snprintf(buf, sizeof(buf), "%.17g", kMaxIdleWatchdogSeconds);
        assert(parse_ok({"--online-mode", "strategy", "--idle-watchdog-s",
                         buf}, out));
        assert(out.idle_watchdog_s == kMaxIdleWatchdogSeconds);
    }
    // rejected lexicon: whitespace / signs outside an exponent / hex /
    // hex-float / nan / inf tokens (2026-10-01: the lex stage now enforces
    // the plain decimal grammar [0-9] '.' 'e'/'E' with one sign allowed
    // only right after an exponent marker -- std::strtod would otherwise
    // accept "0x10"/"0x1p4")
    for (const std::string& bad : {" +1", " -1", "\t-1", "+900", "-0.5",
                                   "1 ", "9 00", "0x10", "0X1P4", "0x1p4",
                                   "abc", "nan", "inf", "0.0e0x"}) {
        OnlineCliOptions probe;
        probe.idle_watchdog_s = -42.5;
        assert(parse_error({"--online-mode", "strategy",
                            "--idle-watchdog-s", bad})
                   .find("plain non-negative decimal") != std::string::npos);
        assert(!parse_ok({"--online-mode", "strategy",
                          "--idle-watchdog-s", bad}, probe));
        assert(probe.idle_watchdog_s == -42.5);
    }
    // rejected: garbage tails / non-finite / ERANGE both directions
    // (2026-10-01) "1e-320" is a legal subnormal literal at the lex
    // stage now (exponent sign allowed); it and "1e309" die at the VALUE
    // stage (ERANGE/non-finite). abc/nan/inf/0.0e0x moved to the lex group
    // above (their letters leave the decimal grammar).
    for (const std::string& bad : {"1e", "1e309", "1e-320"}) {
        assert(parse_error({"--online-mode", "strategy",
                            "--idle-watchdog-s", bad})
                   .find("finite number") != std::string::npos);
    }
    // rejected between the bounds (E26 order): below clock resolution and
    // above the platform cap
    assert(parse_error({"--online-mode", "strategy",
                        "--idle-watchdog-s=1e-10"})
               .find("clock resolution") != std::string::npos);
    assert(parse_error({"--online-mode", "strategy",
                        "--idle-watchdog-s=9.3e9"})
               .find("upper bound") != std::string::npos);
    assert(parse_error({"--online-mode", "strategy",
                        "--idle-watchdog-s=1e308"})
               .find("upper bound") != std::string::npos);
    // missing value
    assert(parse_error({"--online-mode", "strategy", "--idle-watchdog-s"})
               .find("requires a value") != std::string::npos);
    // --idle-* typos stay hard errors (family fail-closed)
    assert(parse_error({"--online-mode", "strategy", "--idle-watchdog=1"})
               .find("unknown online-family") != std::string::npos);

    // R15 (P2-4, 2026-09-28): a value-taking online option never consumes
    // a "--"-prefixed token as its value. The report shape
    // "--bridge-dir --close-input" used to store the directory name
    // "--close-input" AND silently arm the close flag (the token was
    // swallowed, the flag never parsed) -- the Python bridge side then
    // waited on the real directory while the C++ side parked in the wrong
    // one: both sides blocked forever.
    for (const char* opt : {"--bridge-dir", "--command-fifo",
                            "--request-queue-csv"}) {
        assert(parse_error({"--online-mode", "strategy", opt,
                            "--close-input"})
                   .find("looks like another option") != std::string::npos);
        assert(parse_error({"--online-mode", "strategy", opt,
                            "--online-mode"})
                   .find("looks like another option") != std::string::npos);
    }
    // F1-CPP C1-6 (2026-09-29): the "--" value guard covers the INLINE
    // form too. The guard originally sat inside the !has_inline_value
    // branch, so "--bridge-dir=--close-input" was accepted, stored the
    // literal directory name, and -- under the official runner, which
    // cd's to the repo root first -- materialized a "--close-input"
    // directory in the repo root (ensure_bridge_dir) while the runner's
    // no-timeout FIFO wait loop wedged both sides. Same poison as the
    // separated shape, so same diagnosis; the FIFO/CSV siblings are
    // pinned as well even though their downstream existence checks
    // happened to reject them before.
    for (const char* inline_form : {"--bridge-dir=--close-input",
                                    "--command-fifo=--close-input",
                                    "--request-queue-csv=--close-input"}) {
        assert(parse_error({"--online-mode", "strategy", inline_form})
                   .find("looks like another option") != std::string::npos);
    }
    {
        // the failed parse must not leave the inverted close semantics
        // behind in the out struct (no partial write)
        OnlineCliOptions probe;
        probe.close_input = true;
        assert(!parse_ok({"--online-mode", "strategy", "--bridge-dir",
                          "--close-input"},
                         probe));
        assert(probe.close_input);  // untouched by the failed parse
    }
    // the safe separated form still parses
    assert(parse_ok({"--online-mode", "strategy", "--bridge-dir",
                     "/tmp/bridge"},
                    out));
    assert(out.bridge_dir == "/tmp/bridge");

    // R15b (P2-6, 2026-09-28): an explicitly given but EMPTY value is a
    // hard parse error, distinct from the option being absent (the pre-fix
    // shape stored "" -> readability check skipped -> expected_requests 0
    // -> every completion/delivery gate green on a zero-request run).
    assert(parse_error({"--online-mode", "strategy",
                        "--request-queue-csv="})
               .find("non-empty value") != std::string::npos);
    assert(parse_error({"--online-mode", "strategy", "--request-queue-csv",
                        ""})
               .find("non-empty value") != std::string::npos);
    assert(parse_error({"--online-mode", "strategy", "--bridge-dir="})
               .find("non-empty value") != std::string::npos);
    assert(parse_error({"--online-mode", "strategy", "--bridge-dir", ""})
               .find("non-empty value") != std::string::npos);
    assert(parse_error({"--online-mode", "strategy", "--command-fifo="})
               .find("non-empty value") != std::string::npos);

    // R15c (P2-4): --bridge-dir must be a directory when it exists
    // (non-existence stays legal: ensure_bridge_dir creates it);
    // --command-fifo must name an existing FIFO (already exercised in R9).
    assert(parse_error({"--online-mode", "strategy",
                        "--bridge-dir=/proc/self/exe"})
               .find("not a directory") != std::string::npos);
    assert(parse_ok({"--online-mode", "strategy",
                     "--bridge-dir=/tmp/cli_online_no_such_dir"},
                    out));
    assert(out.bridge_dir == "/tmp/cli_online_no_such_dir");

    // R15d: the online entry's argv strip mirrors the parser's boolean
    // value consumption (main_online.cc): the shared lexicon lives in
    // online_boolean_literal, pinned here for both sides.
    assert(online_boolean_literal("true") == 1 &&
           online_boolean_literal("1") == 1 &&
           online_boolean_literal("false") == 0 &&
           online_boolean_literal("0") == 0 &&
           online_boolean_literal("yes") == -1 &&
           online_boolean_literal("") == -1 &&
           online_boolean_literal("TRUE") == -1);

    std::remove(fifo_path.c_str());
    std::printf("[cli] ALL PASS: R1-R16 online CLI contract verified "
                "(incl. B.3 node-gc arm removal, FP1 "
                "hardened unsigned lexicon + watchdog bounds, P2-4/5/6 "
                "value-consumption + unified boolean forms + path "
                "legality, C3 close-input/command-fifo mutual exclusion)\n");
    return 0;
}
