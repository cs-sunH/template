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
  R6  --bridge-dir is parsed and stored (consumed by the online entry:
       the decision bridge's working directory; required in online
       mode).
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
  R15 M5-03 interaction (2026-09-26): the shared CmdLineParser fail-closes
       on unknown options, so the online family must be stripped from argv
       before the shared parser re-parses the static family (the raw
       handover aborted the official online binary with "Option
       'online-mode' does not exist" before the bridge FIFOs existed --
       smoke failure 2026-09-26 round 1 full). strip_online_family_argv
       (a) keeps argv[0] and every static token (values included),
       (b) drops online-family tokens in both value forms -- separate-value
       options also drop their value token exactly the way parse_online_cli
       consumed it (blind argv[++i]), boolean flags drop only themselves,
       (c) drops the WHOLE online family at prefix level (name-level
       unknown detection stays parse_online_cli's job, which runs first).

Build: registered in the CMake build (face M12② same fix) as target
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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/stat.h>

#include "astra-sim/workload/execution_driven/OnlineCli.hh"

using namespace AstraSim::ExecutionDriven;

// The aggregation build uses -O3 -DNDEBUG, which compiles bare CLI_CHECK()
// into no-ops: every check below would silently vanish and main() would
// print ALL PASS regardless of the parse failures (the R1-R14 contract
// coverage would be void in the default build). CLI_CHECK keeps every
// check fail-closed under NDEBUG -- same pattern as
// ingress_idle_fixture.cc's FIXTURE_CHECK: one stderr line, then abort.
#define CLI_CHECK(cond)                                                      \
    ((cond) ? (void)0                                                        \
            : (std::fprintf(stderr, "[cli_online_test] CHECK FAILED: %s "    \
                                    "(%s:%d)\n",                            \
                            #cond, __FILE__, __LINE__), std::abort()))

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
    CLI_CHECK(!parse_online_cli(a.argc, a.argv.data(), out, error));
    return error;
}

// Discard buffer for parse calls whose error text is not under test
// (R15 argv-strip cases assert on the STRIPPED argv, not on the message).
std::string& err_buf() {
    static std::string error;
    error.clear();
    return error;
}

}  // namespace

int main() {
    OnlineCliOptions out;

    // R1: missing --online-mode is a hard error
    CLI_CHECK(parse_error({}).find("--online-mode") != std::string::npos);

    // R2: mode token required
    CLI_CHECK(parse_error({"--online-mode"})
               .find("requires a value") != std::string::npos);
    // R2: valid token, separate-value form
    CLI_CHECK(parse_ok({"--online-mode", "strategy"}, out));
    CLI_CHECK(out.mode == "strategy");
    // R2: valid token, inline "=" form
    CLI_CHECK(parse_ok({"--online-mode=strategy"}, out));
    CLI_CHECK(out.mode == "strategy");
    // R2: the replay token was removed with the replay route (2026-08-18)
    CLI_CHECK(parse_error({"--online-mode", "replay"})
               .find("unknown --online-mode value") != std::string::npos);
    CLI_CHECK(parse_error({"--online-mode=replay"})
               .find("unknown --online-mode value") != std::string::npos);
    // R2: any other token is a hard error
    CLI_CHECK(parse_error({"--online-mode", "bogus"})
               .find("unknown --online-mode value") != std::string::npos);
    CLI_CHECK(parse_error({"--online-mode=offline"})
               .find("unknown --online-mode value") != std::string::npos);

    // R3: --online-mode strategy alone -> request-neutral defaults (IDLE; no
    // CSV, no command FIFO)
    CLI_CHECK(parse_ok({"--online-mode", "strategy"}, out));
    CLI_CHECK(out.mode == "strategy" && !out.close_input &&
           out.bridge_dir.empty() &&
           out.request_queue_csv.empty() && out.command_fifo.empty());
    // 2026-09-05 default pin (R3): the parse path writes back its local
    // default unconditionally, so the BARE-invocation behavioral default of
    // the parking watchdog is armed at 1.0 s (explicit 0 = off, R14).
    CLI_CHECK(out.idle_watchdog_s == 1.0);

    // R4: existing readable CSV accepted
    CLI_CHECK(parse_ok({"--online-mode", "strategy",
                     "--request-queue-csv=/proc/self/exe"}, out));
    CLI_CHECK(out.request_queue_csv == "/proc/self/exe");
    // R4: space-separated value form accepted
    CLI_CHECK(parse_ok({"--online-mode", "strategy", "--request-queue-csv",
                     "/proc/self/exe"}, out));
    CLI_CHECK(out.request_queue_csv == "/proc/self/exe");
    // R4: missing file rejected
    CLI_CHECK(parse_error({"--online-mode", "strategy",
                        "--request-queue-csv=/no/such/file.csv"})
               .find("not readable") != std::string::npos);

    // R5: --close-input flag
    CLI_CHECK(parse_ok({"--online-mode", "strategy", "--close-input"}, out));
    CLI_CHECK(out.close_input);
    // R5: value form rejected
    CLI_CHECK(parse_error({"--online-mode", "strategy", "--close-input=true"})
               .find("no value") != std::string::npos);

    // R6: --bridge-dir parsed and stored
    CLI_CHECK(parse_ok({"--online-mode", "strategy", "--bridge-dir=/tmp/bridge"},
                    out));
    CLI_CHECK(out.bridge_dir == "/tmp/bridge");
    // R6: space-separated form
    CLI_CHECK(parse_ok({"--online-mode", "strategy", "--bridge-dir", "/tmp/bridge"},
                    out));
    CLI_CHECK(out.bridge_dir == "/tmp/bridge");

    // R11: --bridge-timeout-ms parsed and stored (both value forms); default
    // 0 = wait forever; negative / garbage / over-int-range are hard errors.
    CLI_CHECK(parse_ok({"--online-mode", "strategy",
                     "--bridge-timeout-ms=300000"}, out));
    CLI_CHECK(out.bridge_timeout_ms == 300000);
    CLI_CHECK(parse_ok({"--online-mode", "strategy", "--bridge-timeout-ms",
                     "300000"}, out));
    CLI_CHECK(out.bridge_timeout_ms == 300000);
    CLI_CHECK(parse_ok({"--online-mode", "strategy"}, out));
    CLI_CHECK(out.bridge_timeout_ms == 0);
    CLI_CHECK(parse_error({"--online-mode", "strategy",
                        "--bridge-timeout-ms=-1"})
               .find("non-negative integer") != std::string::npos);
    CLI_CHECK(parse_error({"--online-mode", "strategy", "--bridge-timeout-ms=x"})
               .find("non-negative integer") != std::string::npos);
    CLI_CHECK(parse_error({"--online-mode", "strategy",
                        "--bridge-timeout-ms=99999999999"})
               .find("int range") != std::string::npos);

    // R7: typos in the online family are hard errors
    CLI_CHECK(parse_error({"--online-mode", "strategy", "--request-qeue-csv=/x"})
               .find("unknown online-family") != std::string::npos);
    CLI_CHECK(parse_error({"--online-mode", "strategy", "--online-flag"})
               .find("unknown online-family") != std::string::npos);
    CLI_CHECK(parse_error({"--online-mode", "strategy", "--bridge-dirx=/x"})
               .find("unknown online-family") != std::string::npos);

    // R7: unrelated (non-online-family) options are left to the shared
    // CmdLineParser, not rejected here
    CLI_CHECK(parse_ok({"--online-mode", "strategy", "--comm-scale=2.0"}, out));

    // R8: missing value after --bridge-dir / --request-queue-csv /
    // --command-fifo
    CLI_CHECK(parse_error({"--online-mode", "strategy", "--bridge-dir"})
               .find("requires a value") != std::string::npos);
    CLI_CHECK(parse_error({"--online-mode", "strategy", "--request-queue-csv"})
               .find("requires a value") != std::string::npos);
    CLI_CHECK(parse_error({"--online-mode", "strategy", "--command-fifo"})
               .find("requires a value") != std::string::npos);

    // R9 (2026-10-01 起按 R15c 口径修订): --command-fifo parsed and
    // stored (inline and separate-value forms). The value must name an
    // EXISTING FIFO, so the fixture mkfifs one (per-process path).
    const std::string fifo_path =
        "/tmp/cli_online_fixture_" + std::to_string(::getpid()) + ".fifo";
    // 陈旧 FIFO 防护：pid 复用 + 前一次 assert 中止运行残留的 FIFO（main
    // 尾部的 remove 被跳过）会让 mkfifo 撞 EEXIST、把检查变成假红。
    ::unlink(fifo_path.c_str());
    CLI_CHECK(::mkfifo(fifo_path.c_str(), 0600) == 0);
    CLI_CHECK(parse_ok({"--online-mode", "strategy",
                     "--command-fifo=" + fifo_path}, out));
    CLI_CHECK(out.command_fifo == fifo_path);
    CLI_CHECK(parse_ok({"--online-mode", "strategy", "--command-fifo",
                     fifo_path}, out));
    CLI_CHECK(out.command_fifo == fifo_path);
    // R15c: 缺席路径与存在但非 FIFO 的文件都是硬错误。
    CLI_CHECK(parse_error({"--online-mode", "strategy",
                        "--command-fifo=/no/such/fixture_cmd.fifo"})
               .find("must name an existing FIFO") != std::string::npos);
    CLI_CHECK(parse_error({"--online-mode", "strategy",
                        "--command-fifo=/proc/self/exe"})
               .find("must name an existing FIFO") != std::string::npos);
    // R7: --command-* typos are hard errors
    CLI_CHECK(parse_error({"--online-mode", "strategy", "--command-fifoX=/x"})
               .find("unknown online-family") != std::string::npos);

    // R16（2026-10-01 补齐）：--close-input 与 --command-fifo 互斥——启动
    // 即关门会在 FIFO 生产者注入任何东西之前杀死通道，无 CSV 的 run 以
    // 零请求绿色退出。本仓 --close-input 是冻结的裸旗标词法（带值形态
    // 一律硬错"takes no value"），故只测裸旗标形态 + FIFO 的互斥。
    CLI_CHECK(parse_error({"--online-mode", "strategy",
                        "--command-fifo=" + fifo_path, "--close-input"})
               .find("mutually exclusive") != std::string::npos);
    CLI_CHECK(parse_error({"--online-mode", "strategy",
                        "--command-fifo=" + fifo_path, "--close-input=true"})
               .find("takes no value") != std::string::npos);
    // R16b（R2 复核补齐）：--request-queue-csv 与 --command-fifo 互斥
    // （CSV 臂持有 queue-index 域，FIFO 信封 queue_index 恒 -1）。
    CLI_CHECK(parse_error({"--online-mode", "strategy",
                        "--request-queue-csv=/proc/self/exe",
                        "--command-fifo=" + fifo_path})
               .find("mutually exclusive") != std::string::npos);
    ::unlink(fifo_path.c_str());

    // missing --online-mode even with other online flags
    CLI_CHECK(parse_error({"--request-queue-csv=/proc/self/exe", "--close-input"})
               .find("--online-mode") != std::string::npos);

    // R11: phase-7 §10.4 --request-max-arrival-ns
    // (default: 0 = UNBOUNDED arrival window -- backport fix
    // 2026-08-16, sh_2.0测试 §5.1: the old 30e9 default burned the 30s
    // acceptance window into the code and silently dropped over-window
    // requests; the bound is now explicit-only and its drops fail-close
    // the run-end completion audit)
    CLI_CHECK(parse_ok({"--online-mode", "strategy"}, out));
    CLI_CHECK(out.request_max_arrival_ns == 0);
    CLI_CHECK(parse_ok({"--online-mode", "strategy",
                     "--request-max-arrival-ns=12345"}, out));
    CLI_CHECK(out.request_max_arrival_ns == 12345);
    // R11: non-integer / negative values are hard errors
    CLI_CHECK(parse_error({"--online-mode", "strategy",
                        "--request-max-arrival-ns=-5"})
               .find("non-negative integer") != std::string::npos);
    // R11: missing value forms rejected
    CLI_CHECK(parse_error({"--online-mode", "strategy",
                        "--request-max-arrival-ns"})
               .find("requires a value") != std::string::npos);
    // R11: typos in the new family are hard errors
    CLI_CHECK(parse_error({"--online-mode", "strategy",
                        "--request-window-row=64"})
               .find("unknown online-family") != std::string::npos);

    // R10: --sensing-enabled default OFF (feature flag; phase 6 before 默认关)
    CLI_CHECK(parse_ok({"--online-mode", "strategy"}, out));
    CLI_CHECK(!out.sensing_enabled);
    // R10: flag form enables
    CLI_CHECK(parse_ok({"--online-mode", "strategy", "--sensing-enabled"}, out));
    CLI_CHECK(out.sensing_enabled);
    // R10: value form rejected
    CLI_CHECK(parse_error({"--online-mode", "strategy", "--sensing-enabled=true"})
               .find("takes no value") != std::string::npos);
    // R10: --sensing-* typos are hard errors
    CLI_CHECK(parse_error({"--online-mode", "strategy", "--sensing-enable=/x"})
               .find("unknown online-family") != std::string::npos);
    CLI_CHECK(parse_error({"--online-mode", "strategy", "--sensing-x"})
               .find("unknown online-family") != std::string::npos);
    // R10: coexists with the rest of the family
    CLI_CHECK(parse_ok({"--online-mode", "strategy", "--close-input",
                     "--sensing-enabled", "--bridge-dir=/tmp/bridge"}, out));
    CLI_CHECK(out.sensing_enabled && out.close_input &&
           out.bridge_dir == "/tmp/bridge");

    // R12 B.3 cleanup (2026-09-05): the --online-node-gc arm was removed;
    // M2 node GC is always on. The literal flag (both value forms) is an
    // unknown online-family option -- stale scripts fail closed at parse
    // time instead of silently disabling collection.
    CLI_CHECK(parse_error({"--online-mode", "strategy", "--online-node-gc=0"})
               .find("unknown online-family") != std::string::npos);
    CLI_CHECK(parse_error({"--online-mode", "strategy", "--online-node-gc", "1"})
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
            CLI_CHECK(parse_error({"--online-mode", "strategy", opt, bad})
                       .find("integer") != std::string::npos);
            // separate-value form reaches the same lexicon
            CLI_CHECK(!parse_ok({"--online-mode", "strategy", opt, bad}, probe));
            CLI_CHECK(probe.request_max_arrival_ns == 777 &&
                   probe.bridge_timeout_ms == 777);
        }
    }
    // R13: int-typed options bound at exactly INT_MAX.
    CLI_CHECK(parse_ok({"--online-mode", "strategy",
                     "--bridge-timeout-ms=2147483647"}, out));
    CLI_CHECK(out.bridge_timeout_ms == 2147483647);
    CLI_CHECK(parse_error({"--online-mode", "strategy",
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
        CLI_CHECK(parse_ok({"--online-mode", "strategy", "--online-validate=0"},
                        out));
        CLI_CHECK(out.online_validate == 0);
        CLI_CHECK(parse_ok({"--online-mode", "strategy", "--online-validate=1"},
                        out));
        CLI_CHECK(out.online_validate == 1);
        OnlineCliOptions probe;
        CLI_CHECK(parse_ok({"--online-mode", "strategy", "--online-validate",
                         "0"},
                        probe));
        CLI_CHECK(probe.online_validate == 0);
        CLI_CHECK(parse_ok({"--online-mode", "strategy", "--online-validate",
                         "1"},
                        probe));
        CLI_CHECK(probe.online_validate == 1);
        for (const std::string& bad :
             {"2", "3", "01", "on", "true", "-1", "+1", "1x", " 1",
              "2147483647", "2147483648", "4294967296"}) {
            OnlineCliOptions p = sentinel;
            CLI_CHECK(parse_error({"--online-mode", "strategy",
                                "--online-validate=" + bad})
                       .find("unknown --online-validate value") !=
                   std::string::npos);
            // separate-value form reaches the same enum check
            CLI_CHECK(!parse_ok(
                {"--online-mode", "strategy", "--online-validate", bad}, p));
            CLI_CHECK(p.online_validate == 777);
        }
    }

    // R14: FP1 watchdog lexicon and E26 bounds ordering.
    // 2026-09-05 default pin: the struct default is armed at 1.0 s; the
    // explicit 0 = off escape is asserted immediately below.
    {
        OnlineCliOptions defaults_probe;
        CLI_CHECK(defaults_probe.idle_watchdog_s == 1.0);
    }
    // accepted: 0 = off, sub-second, both bounds themselves
    CLI_CHECK(parse_ok({"--online-mode", "strategy", "--idle-watchdog-s", "0"},
                    out));
    CLI_CHECK(out.idle_watchdog_s == 0.0);
    CLI_CHECK(parse_ok({"--online-mode", "strategy", "--idle-watchdog-s=0.5"},
                    out));
    CLI_CHECK(out.idle_watchdog_s == 0.5);
    {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.17g", kMinIdleWatchdogSeconds);
        CLI_CHECK(parse_ok({"--online-mode", "strategy", "--idle-watchdog-s",
                         buf}, out));
        CLI_CHECK(out.idle_watchdog_s == kMinIdleWatchdogSeconds);
        std::snprintf(buf, sizeof(buf), "%.17g", kMaxIdleWatchdogSeconds);
        CLI_CHECK(parse_ok({"--online-mode", "strategy", "--idle-watchdog-s",
                         buf}, out));
        CLI_CHECK(out.idle_watchdog_s == kMaxIdleWatchdogSeconds);
    }
    // rejected lexicon: whitespace / sign characters anywhere
    for (const std::string& bad : {" +1", " -1", "\t-1", "+900", "-0.5",
                                   "1 ", "9 00"}) {
        OnlineCliOptions probe;
        probe.idle_watchdog_s = -42.5;
        CLI_CHECK(parse_error({"--online-mode", "strategy",
                            "--idle-watchdog-s", bad})
                   .find("whitespace or sign") != std::string::npos);
        CLI_CHECK(!parse_ok({"--online-mode", "strategy",
                          "--idle-watchdog-s", bad}, probe));
        CLI_CHECK(probe.idle_watchdog_s == -42.5);
    }
    // rejected: garbage tails / non-finite / ERANGE both directions
    for (const std::string& bad : {"abc", "1e", "nan", "inf", "1e309",
                                   "1e-320", "0.0e0x"}) {
        CLI_CHECK(parse_error({"--online-mode", "strategy",
                            "--idle-watchdog-s", bad})
                   .find("finite number") != std::string::npos);
    }
    // rejected between the bounds (E26 order): below clock resolution and
    // above the platform cap
    CLI_CHECK(parse_error({"--online-mode", "strategy",
                        "--idle-watchdog-s=1e-10"})
               .find("clock resolution") != std::string::npos);
    CLI_CHECK(parse_error({"--online-mode", "strategy",
                        "--idle-watchdog-s=9.3e9"})
               .find("upper bound") != std::string::npos);
    CLI_CHECK(parse_error({"--online-mode", "strategy",
                        "--idle-watchdog-s=1e308"})
               .find("upper bound") != std::string::npos);
    // missing value
    CLI_CHECK(parse_error({"--online-mode", "strategy", "--idle-watchdog-s"})
               .find("requires a value") != std::string::npos);
    // --idle-* typos stay hard errors (family fail-closed)
    CLI_CHECK(parse_error({"--online-mode", "strategy", "--idle-watchdog=1"})
               .find("unknown online-family") != std::string::npos);

    // R15 (M5-03 interaction, 2026-09-26): strip_online_family_argv leaves
    // ONLY the shared static family. The shape below is the official
    // runner's invocation (run_online_strategy.sh) that used to abort the
    // binary in the shared parser before the bridge FIFOs existed.
    {
        const std::vector<std::string> toks = {
            // (no argv[0] here: Args prepends "fake_online_bin")
            "--online-mode", "strategy",            // separate value form
            "--bridge-dir=/tmp/b",                  // inline value form
            "--online-validate", "0",
            "--bridge-timeout-ms", "120000",
            "--request-queue-csv", "/proc/self/exe",  // must be readable (R4)
            "--close-input",                        // boolean flag
            "--sensing-enabled",                    // boolean flag
            "--workload-configuration", "/tmp/et_prefix",
            "--comm-group-configuration=/tmp/cg.json",
            "--system-configuration", "/tmp/system.json",
            "--remote-memory-configuration", "/tmp/rm.json",
            "--network-configuration=/tmp/net.yml",
            "--logging-folder=off",
            "--metrics-configuration=/tmp/m.json",
            "--metrics-detail=full",
        };
        Args a(toks);
        // The strip runs AFTER a successful parse in the real binary; pin
        // that this exact argv parses (guards the test itself against
        // drifting away from a legal invocation).
        OnlineCliOptions parsed;
        CLI_CHECK(parse_online_cli(a.argc, a.argv.data(), parsed, err_buf()));
        const auto stripped = strip_online_family_argv(a.argc, a.argv.data());
        const std::vector<std::string> expect = {
            "fake_online_bin",
            "--workload-configuration", "/tmp/et_prefix",
            "--comm-group-configuration=/tmp/cg.json",
            "--system-configuration", "/tmp/system.json",
            "--remote-memory-configuration", "/tmp/rm.json",
            "--network-configuration=/tmp/net.yml",
            "--logging-folder=off",
            "--metrics-configuration=/tmp/m.json",
            "--metrics-detail=full",
        };
        CLI_CHECK(stripped.size() == expect.size());
        for (size_t i = 0; i < expect.size() && i < stripped.size(); ++i) {
            CLI_CHECK(std::string(stripped[i]) == expect[i]);
        }
        // (b) exact-mirror value consumption: the value token following a
        // separate-value online option is dropped even when it itself looks
        // like a flag -- parse_online_cli consumed it blind (argv[++i]), so
        // the strip must too, or the two passes would disagree on ownership.
        {
            const std::vector<std::string> pair = {
                "--bridge-dir", "--close-input", "x"};
            Args p(pair);
            const auto s = strip_online_family_argv(p.argc, p.argv.data());
            CLI_CHECK(s.size() == 2);  // argv[0] + "x"
            CLI_CHECK(std::string(s[1]) == "x");
        }
        // (c) prefix-level drop: a name-level UNKNOWN online token is still
        // family (parse_online_cli rejects it before the strip ever runs in
        // the real binary; here it just documents the boundary).
        {
            const std::vector<std::string> typo = {
                "--online-flag=1",
                "--workload-configuration", "/w"};
            Args t(typo);
            const auto s = strip_online_family_argv(t.argc, t.argv.data());
            CLI_CHECK(s.size() == 3);
            CLI_CHECK(std::string(s[1]) == "--workload-configuration");
            CLI_CHECK(std::string(s[2]) == "/w");
        }
        // Degenerate shapes must not overrun: trailing valueless online
        // option, and the empty-argv guard.
        {
            const std::vector<std::string> tail = {
                "--idle-watchdog-s"};
            Args tl(tail);
            const auto s = strip_online_family_argv(tl.argc, tl.argv.data());
            CLI_CHECK(s.size() == 1);
            CLI_CHECK(std::string(s[0]) == "fake_online_bin");
        }
        {
            char* empty_argv[1] = {const_cast<char*>("fake_online_bin")};
            const auto s = strip_online_family_argv(1, empty_argv);
            CLI_CHECK(s.size() == 1);
            CLI_CHECK(std::string(s[0]) == "fake_online_bin");
        }
    }

    std::printf("[cli] ALL PASS: R1-R15 online CLI contract verified "
                "(incl. B.3 node-gc arm removal, FP1 "
                "hardened unsigned lexicon + watchdog bounds, M5-03 "
                "argv strip)\n");
    return 0;
}
