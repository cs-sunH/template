/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

OnlineCli -- execution-driven mechanism layer (wscllm phase 1).

Online entry CLI contract (方案 §4 步骤 1-2 操作 5). The shared CmdLineParser
fail-closes on unknown options since M5-03 (2026-09-26;
CmdLineParser 的 unknown-option fail-close), so the two parsers split argv by ownership: the
online entry parses its own family explicitly and STRICTLY here
(parse_online_cli: unknown online-family names are hard errors), then strips
the family from argv (strip_online_family_argv) before the shared parser
re-parses only the static family -- handing the raw argv over made the
shared parser abort on the first online token ("Option 'online-mode' does
not exist", smoke failure 2026-09-26, round 1 full):

  --online-mode=<mode>  required for the online binary (missing = hard error).
                        <mode> is the token "strategy" (live scheduler;
                        replay was removed with the replay route on
                        2026-08-18). Inline "=" values are legal
                        (--online-mode=strategy); a separate value token is
                        also accepted. Any other token is a hard error.
  --request-queue-csv   optional path of the 8-column request queue CSV.
                        Request-neutral default: when absent the service
                        stays IDLE and reads no pre-loaded queue (the static
                        baseline config, trace_config.csv key
                        request_queue_csv, stays separate from the online
                        defaults).
  --request-max-arrival-ns
                        optional simulation input window upper bound (ns)
                        for turn-0 arrivals; rows beyond it are rejected
                        (never submitted, counted and reported). Default 0 =
                        UNBOUNDED -- the window is an explicit experiment
                        knob only (the pre-2026-08-16 default
                        30,000,000,000 burned the 20.csv first-30-seconds
                        input boundary into the code and silently dropped
                        over-window requests; the official entry passes no
                        window, so the rejection counter reads 0 there).
  --command-fifo        optional FIFO path read by an external producer
                        thread (step 1-10 IDLE fixture). The FIFO must
                        ALREADY EXIST at parse time (mkfifo it before the
                        binary starts) and be a real FIFO; a missing path
                        or a non-FIFO file is a hard parse error
                        (2026-10-01: the reader thread's old soft
                        "cannot open" log left the input gate silently
                        open for the whole run). Mutually exclusive with
                        --close-input (startup close would kill the FIFO
                        channel) and with --request-queue-csv (the CSV
                        path owns the queue-index domain). JSON lines:
                        {"kind":"Submit", session_id, turn_index, request_id,
                        prefill_length, decode_length, arrival_world_ns,
                        inter_request_interval_ns},
                        {"kind":"CloseInput"} (explicit close),
                        {"kind":"EndOfFile"} (EOF terminal command) or
                        {"kind":"Error"} (fail-closed abort; never a silent
                        close -- 合同② 三态区分, phase-7 §10.7).
                        The producer only writes the thread-safe bounded
                        ingress command queue (合同②); the decision bridge
                        stays the decision channel only.
  --bridge-dir          required in online mode: the decision bridge's
                        working directory. The online entry derives the
                        bridge FIFO paths and the remote-memory transaction
                        log location from it (main_online.cc) and aborts
                        fail-closed when it is missing or empty.
  --bridge-timeout-ms   optional decision-bridge response-wait poll timeout
                        in milliseconds (0 = wait forever, the frozen
                        default). When > 0, a Python decision side stalled
                        longer than the timeout aborts the run fail-closed
                        with the standard Python-side-died diagnostics
                        (long-run wedge watchdog, 2026-08-22
                        wedge-diagnosis recommendation; the value must exceed both the
                        workload's worst single-decision time and the
                        Python-side startup FIFO-open wait -- practical
                        lower bound is seconds, suggest >= 10000;
                        opt-in only). P0-2 (2026-08-31, 总文档 §4 P0-2.4)
                        made the official runner default 120000 with an
                        explicit BRIDGE_TIMEOUT_MS=0 escape hatch (wait
                        forever), and fixed the scope label: this timeout
                        only arms the two bridge response polls
                        (DecisionBridge.cc); it does NOT cover the
                        main-loop wait_for_work() parking family (that is
                        --idle-watchdog-s below).
  --idle-watchdog-s     P0-2 (2026-08-31, 总文档 §4 P0-2.3) wall-clock
                        event-loop parking watchdog, double seconds.
                        Default 1.0 (2026-09-05: the watchdog ships armed
                        -- silent stalls fail closed in ~1 s instead of
                        hanging; healthy official runs never reach the
                        parking point since every calendar arrival is
                        pre-scheduled as a queue event). The explicit
                        value 0 = OFF restores the original unbounded
                        wait_for_work() contract (fixtures and
                        external-producer runs that legitimately idle
                        pass it). When > 0, the main
                        loop's input-open parking point
                        (ServiceCoordinator::checked_wait_deadline +
                        wait_for_work_until; steady_clock -- WALL clock,
                        never the simulation clock: real-time traces
                        space turn arrivals hours apart, so simulation time
                        says nothing about liveness) aborts fail-closed
                        after the deadline with the same dead-end
                        diagnostics as the input-open fail-loud branch.
                        FP1 (2026-09-01, sync-A16 batch P; E26): parse-time
                        bounds -- 0 accepted (off); (0, 1e-9) rejected
                        ("below clock resolution"); > 1e9 rejected
                        (kMaxIdleWatchdogSeconds). Strict decimal lexicon
                        (2026-09-29): the token must be a plain decimal
                        number -- digits with an optional fraction and an
                        optional exponent; no whitespace and no sign
                        character anywhere, and the hexadecimal-float
                        ("0x10" = 16.0), inf and nan shapes strtod would
                        otherwise accept are rejected at the lexicon.
                        Covers every silent-stall family the semantic
                        dead-end predicate cannot name, complementing it; the
                        practical lower bound is the largest LEGAL idle
                        gap (real-time campaigns: >= 600 s, overridable
                        per campaign).
  --close-input         mark external production closed once the
                        --request-queue-csv input is fully drained: the
                        close is linearized at the exact finite-input
                        boundary (immediately at startup when no CSV is
                        given or the queue is already at EOF, otherwise at
                        the first main-loop point where the reader reached
                        EOF), so no CSV Submit can ever be rejected by an
                        already-closed ingress. Without the flag the
                        service stays ACTIVE waiting for external
                        production (command-FIFO producer); a silent stall
                        at that wait fails closed in ~1 s through the
                        armed idle watchdog (--idle-watchdog-s, default
                        1.0) instead of hanging forever -- the flag is how
                        a finite CSV run declares "nothing further is
                        coming".
  --sensing-enabled     phase-3 perception feature flag (方案 §6.1/§6.2:
                        感知开关必须经显式 feature flag 进入,阶段 6 前默认关).
                        When set, the tick-end gate computes the per-rank
                        injected-unfinished ledger summary (contract ⑥:
                        compute ops / comm bytes / estimated remaining
                        service / resource state, traceable per
                        request/stage/generation) and delivers it in the
                        StateDelta ledger_summary field. Query/audit data
                        only -- the wscllm strategy's red-line decision
                        inputs (Python queue ledger + KV ledger + static
                        route) never consume it, so sensing-on runs keep the
                        phase-2 decision sequence byte-identical.
  --online-validate     C1 validate switch (2026-08-28), amended by the B.2
                        cleanup (2026-09-05): strict <0|1> enum. 1 (frozen
                        CLI default, fail-closed like the pre-C1 behavior)
                        = every batch runs
                        GraphBatchCommitter::validate_and_commit() (its
                        mandatory pre-commit validation phase); 0 =
                        production fast path, validation
                        skipped entirely (the official runner passes
                        "${SH_ONLINE_VALIDATE:-0}"; smoke/fixture/verify
                        runs pass 1 explicitly) while commit()'s mandatory
                        liveness preflight stays armed. The former N >= 2
                        sample-every-Nth-batch tier is removed: any other
                        token is a hard parse error, so a stale script can
                        never silently change validation frequency.

Hard-error rules: unknown flags in the online family (prefixes --request-,
--bridge-, --close-, --online-, --command-, --sensing-, --idle-) are rejected
here; after this parse succeeds, main_online.cc strips the whole family from
argv before the shared CmdLineParser re-parses the static family (M5-03:
that parser fail-closes on unknown options, so each parser must only ever
see its own family). --request-queue-csv must name an existing, readable
file.
*******************************************************************************/

#ifndef EXECUTION_DRIVEN_ONLINECLI_HH
#define EXECUTION_DRIVEN_ONLINECLI_HH

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace AstraSim {
namespace ExecutionDriven {

/// FP1 (2026-09-01, sync-A16 batch P; contract E25/E26): --idle-watchdog-s
/// numeric bounds, shared by the parser and the CLI unit tests so both judge
/// the same tokens identically. Upper cap 1e9 s is the target-platform
/// contract (Linux x86-64 steady_clock ns has ~9.22e9 s of forward range;
/// the 900 s campaign arming sits far below it). Lower bound 1e-9 s = one
/// clock tick: anything smaller is rejected at parse time as "below clock
/// resolution" so a token that parses can never degenerate into a zero
/// duration at runtime. 0 = OFF is accepted before either bound applies
/// (E26 ordering: lexicon/finiteness -> ==0 accept (off) -> (0, 1e-9)
/// reject -> >1e9 reject).
inline constexpr double kMaxIdleWatchdogSeconds = 1.0e9;
inline constexpr double kMinIdleWatchdogSeconds = 1.0e-9;

struct OnlineCliOptions {
    // Mode token: "" (unset) or "strategy". "" is rejected at the end of
    // parse_online_cli; "replay" is rejected since the path-2 removal
    // (2026-08-18).
    std::string mode;
    bool close_input = false;
    std::string bridge_dir;
    // Decision-bridge response-wait poll timeout (ms; 0 = wait forever, the
    // frozen default). Long-run wedge watchdog, opt-in only (2026-08-22
    // wedge-diagnosis recommendation). P0-2 (2026-08-31): the official
    // runner now defaults BRIDGE_TIMEOUT_MS to 120000; scope = bridge
    // response polls only (see the --bridge-timeout-ms doc above).
    int bridge_timeout_ms = 0;
    // P0-2 (2026-08-31, 总文档 §4 P0-2.3): wall-clock event-loop parking
    // watchdog (seconds). Default 1.0 since 2026-09-05 (ships armed;
    // explicit 0 = off restores the unbounded wait_for_work() contract
    // for fixtures/external-producer runs). See --idle-watchdog-s
    // above; consumed by the main loop's input-open parking point.
    double idle_watchdog_s = 1.0;
    std::string request_queue_csv;
    // Step 1-10: optional external-producer FIFO (IDLE fixture injection).
    std::string command_fifo;
    // Phase-3 perception feature flag (default OFF until phase 6): enables
    // the per-rank injected-unfinished ledger summary delivery (查询/审计
    // 输入,非策略判据输入 -- 感知开关必须经显式 feature flag 进入).
    bool sensing_enabled = false;
    // Phase 7 §10.4: turn-0 arrival upper bound; beyond = rejected (ns).
    // Backport fix (2026-08-16, sh_2.0测试 §5.1): default UNBOUNDED (0 = no
    // cap). The window is an explicit experiment knob only; any drop it
    // causes is counted and fail-closes the run-end completion audit.
    uint64_t request_max_arrival_ns = 0;
    // C1 validate switch (2026-08-28), B.2 cleanup (2026-09-05): strict
    // <0|1> enum -- 1 = full pre-commit validation (the pre-C1 behavior,
    // frozen CLI default so bare invocations stay fail-closed); 0 =
    // production skip (commit() retains its mandatory liveness preflight).
    // The N >= 2 sample-every-Nth-batch tier is removed; any other value is
    // rejected at parse time. The official runner passes
    // "${SH_ONLINE_VALIDATE:-0}".
    int online_validate = 1;
};

/// Parse argv for the online family. Returns false and fills `error` on any
/// contract violation; leaves `out` untouched on failure.
bool parse_online_cli(int argc, char* argv[], OnlineCliOptions& out,
                      std::string& error);

/// True when the token names an online-family option (with or without an
/// inline "=" value): any "--name" under the prefixes --request- / --bridge-
/// / --close- / --online- / --command- / --sensing- / --idle- that
/// parse_online_cli owns. Exported (not an anonymous-namespace helper) so
/// main_online.cc and the argv strip below share ONE family-boundary
/// definition with the parser (W2 lesson: paired code must not evolve
/// single-sidedly).
bool is_online_family_option(const std::string& token);

/// True when the named online-family option consumes a separate value token
/// (the "--name value" shape; inline "=" forms never do). Exact mirror of
/// the value-taking branches in parse_online_cli; exported with
/// is_online_family_option so the argv strip and the parser cannot drift.
bool online_family_option_takes_value(const std::string& name);

/// Build a stripped copy of argv holding ONLY the shared static family:
/// argv[0] plus every token is_online_family_option rejects, with the
/// separate value token of each value-taking online option removed too
/// (exactly the tokens parse_online_cli consumed via argv[++i]). The
/// elements alias the original argv storage (no string copies; lifetime =
/// the caller's argv). Call ONLY after parse_online_cli succeeded: name-
/// level unknown-option detection stays its job, this strip is prefix-level.
/// Reason (M5-03, 2026-09-26): the shared CmdLineParser fail-closes on
/// unknown options; the online entry used to hand the raw argv over and the
/// parser aborted on the first online token before the bridge FIFOs existed.
std::vector<char*> strip_online_family_argv(int argc, char* argv[]);

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_ONLINECLI_HH
