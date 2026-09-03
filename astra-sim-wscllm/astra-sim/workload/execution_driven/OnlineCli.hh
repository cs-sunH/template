/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

OnlineCli -- execution-driven mechanism layer (wscllm phase 1).

Online entry CLI contract (方案 §4 步骤 1-2 操作 5). The shared CmdLineParser
uses cxxopts with allow_unrecognised_options() (CmdLineParser.cc:18-21):
unknown options are silently ignored with no strong-typed values, so the
online entry must parse its own family explicitly:

  --online-mode=<mode>  required for the online binary (missing = hard error).
                        <mode> is the token "strategy" (live scheduler;
                        replay was removed with the replay route on
                        2026-08-18). Inline "=" values are legal
                        (--online-mode=strategy); a separate value token is
                        also accepted. Any other token is a hard error.
  --request-queue-csv   optional path of the 8-column request queue CSV.
                        Request-neutral default: when absent the service
                        stays IDLE and reads no pre-loaded queue (the static
                        baseline config, trace_config.csv:12, stays separate
                        from the online defaults).
  --request-window-rows advisory only since the P0 turn-0 fix (2026-08-30):
                        the reader builds a full turn-0 arrival calendar
                        from a single streaming index pass and submits in
                        ARRIVAL order regardless of this value (the old
                        semantics -- bounding how far ahead rows could be
                        READ, which tied turn-0 discovery to file position --
                        is exactly the late-discovery defect the fix
                        eradicates: early-arriving turn-0 rows sitting late
                        in the file were submitted after their declared
                        arrival). The knob is still parsed, stored, reported
                        and checkpointed ("calendar reader: advisory"), 0
                        included, for CLI/checkpoint compatibility; it no
                        longer changes any behavior.
  --request-max-arrival-ns
                        frozen simulation input window upper bound (ns) for
                        turn-0 arrivals; rows beyond it are rejected (never
                        submitted, counted and reported). Frozen default is
                        30,000,000,000 (the 20.csv first-30-seconds input
                        boundary; the input's own max arrival is 25.96s, so
                        the rejection counter reads 0 on the allowed input).
  --command-fifo        optional FIFO path read by an external producer
                        thread (step 1-10 IDLE fixture). JSON lines:
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
  --bridge-dir          reserved for the step-1-7 decision bridge; parsed and
                        stored but unused in step 1-2.
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
                        event-loop parking watchdog, double seconds,
                        frozen default 0 = OFF (the original unbounded
                        wait_for_work() contract). When > 0, the main
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
                        (kMaxIdleWatchdogSeconds); no whitespace and no
                        sign character anywhere in the token (strtod would
                        otherwise accept " +1"/" -1" shapes and an
                        underflowing subnormal could degenerate to a
                        zero duration misread as "off"). Covers every
                        silent-stall family the semantic dead-end
                        predicate cannot name, complementing it; the
                        practical lower bound is the largest LEGAL idle
                        gap (real-time campaigns: >= 600 s, overridable
                        per campaign).
  --close-input         mark the input closed after the CSV is fully drained
                        (otherwise the service stays ACTIVE waiting for the
                        step-1-7 bridge).
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
  --online-node-gc      M2 node GC (2026-08-23; A1 amortization 2026-08-28):
                        value <0|1>, frozen default 1 (on -- A1 flipped the
                        2026-08-23 ruling: collection is amortized, the
                        commit tail only counts pending candidates and
                        drains every per-rank NodeStore once >= 4096 have
                        accumulated, then the run end forces one final
                        collection, eliminating the light-load wall regression
                        that motivated the old default-off). The commit tail
                        also prunes the (rank, json id) -> store id map at
                        the same watermark, keeping the C++ side at the
                        in-flight window instead of the whole-run cumulative
                        graph (memory). Collected nodes were finished, so
                        edges that still reference them resolve to the
                        NodeStore dead-parent no-op; validate() resolves
                        pruned ids below the per-rank dense-prefix watermark
                        only -- never-emitted ids stay fail-closed. 0 =
                        pre-M2 behavior (nodes never erased; emergency
                        rollback arm). Inline "=" and separate-value forms
                        legal; any other token is a hard error.
  --online-validate     C1 validate switch (2026-08-28): <0|1|N>. 1 (frozen
                        CLI default, fail-closed like the pre-C1 behavior)
                        = every batch runs GraphBatchCommitter::validate()
                        before commit; 0 = production fast path, validation
                        skipped entirely (the official runner passes
                        "${SH_ONLINE_VALIDATE:-0}"; smoke/fixture/verify
                        runs pass 1 explicitly); N >= 2 = sample every Nth
                        committed batch (batch index % N == 0 -- the counter
                        is the committer's graph_batch_count, i.e. the batch
                        about to be committed). Sampling changes only the
                        validate counters/diagnostics (graph_validate_ns /
                        graph_validate_count and the consistency record's
                        validate fields -- whitelisted cpp.log diffs), never
                        the committed state: commit() is untouched and a
                        validated batch still gates on validate() first.

Hard-error rules: unknown flags in the online family (prefixes --request-,
--bridge-, --close-, --online-) are rejected, because the shared parser would
otherwise swallow typos silently. --request-queue-csv must name an existing,
readable file.
*******************************************************************************/

#ifndef EXECUTION_DRIVEN_ONLINECLI_HH
#define EXECUTION_DRIVEN_ONLINECLI_HH

#include <cstddef>
#include <cstdint>
#include <string>

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
    // watchdog (seconds; 0 = off, the frozen default). See --idle-watchdog-s
    // above; consumed by the main loop's input-open parking point.
    double idle_watchdog_s = 0.0;
    std::string request_queue_csv;
    // Step 1-10: optional external-producer FIFO (IDLE fixture injection).
    std::string command_fifo;
    // Phase-3 perception feature flag (default OFF until phase 6): enables
    // the per-rank injected-unfinished ledger summary delivery (查询/审计
    // 输入,非策略判据输入 -- 感知开关必须经显式 feature flag 进入).
    bool sensing_enabled = false;
    // Phase 7 §10.4 / P0 fix (2026-08-30): ADVISORY ONLY. The turn-0
    // arrival calendar is always complete and submissions always follow
    // arrival order; this knob no longer bounds discovery (the old row
    // window semantics was the late-discovery defect). Kept parsed,
    // reported and checkpointed (default 128, 0 accepted) for CLI and
    // checkpoint compatibility.
    size_t request_window_rows = 128;
    // Phase 7 §10.4: turn-0 arrival upper bound; beyond = rejected (ns).
    // Backport fix (2026-08-16, sh_2.0测试 §5.1): default UNBOUNDED (0 = no
    // cap). The window is an explicit experiment knob only; any drop it
    // causes is counted and fail-closes the run-end completion audit.
    uint64_t request_max_arrival_ns = 0;
    // M2 node GC (2026-08-23; A1 amortization 2026-08-28): 1 = collect
    // finished childless nodes at the committer's end-of-commit quiescent
    // point (amortized: the commit tail counts pending candidates and only
    // drains once >= 4096 accumulated; the run end forces one final pass);
    // 0 = pre-M2 never-erase behavior. Frozen default 1 (A1 flipped the
    // 2023-08-23 light-load-regression ruling; see main_online.cc).
    int online_node_gc = 1;
    // C1 validate switch (2026-08-28): 1 = full pre-commit validation (the
    // pre-C1 behavior, frozen CLI default so bare invocations stay
    // fail-closed); 0 = production skip; N >= 2 = sample every Nth batch.
    // The official runner passes "${SH_ONLINE_VALIDATE:-0}".
    int online_validate = 1;
};

/// Parse argv for the online family. Returns false and fills `error` on any
/// contract violation; leaves `out` untouched on failure.
bool parse_online_cli(int argc, char* argv[], OnlineCliOptions& out,
                      std::string& error);

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_ONLINECLI_HH
