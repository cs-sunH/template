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
  --request-window-rows optional WindowedTraceReader high watermark (phase 7
                        §10.4): max un-consumed rows read ahead of the
                        simulation. 0 = unbounded (one pump reads the whole
                        file: the full-pass control arm of the window
                        benchmark); frozen default for production runs is
                        128 (the 20.csv input's max consecutive same-session
                        span is 72). Window semantics never change the
                        decision sequence -- only when rows leave the disk.
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

struct OnlineCliOptions {
    // Mode token: "" (unset) or "strategy". "" is rejected at the end of
    // parse_online_cli; "replay" is rejected since the path-2 removal
    // (2026-08-18).
    std::string mode;
    bool close_input = false;
    std::string bridge_dir;
    std::string request_queue_csv;
    // Step 1-10: optional external-producer FIFO (IDLE fixture injection).
    std::string command_fifo;
    // Phase-3 perception feature flag (default OFF until phase 6): enables
    // the per-rank injected-unfinished ledger summary delivery (查询/审计
    // 输入,非策略判据输入 -- 感知开关必须经显式 feature flag 进入).
    bool sensing_enabled = false;
    // Phase 7 §10.4: WindowedTraceReader high watermark (0 = unbounded).
    // Frozen production default 128; the 20.csv window benchmark sweeps
    // {0, 8, 16, 32, 64, 128}.
    size_t request_window_rows = 128;
    // Phase 7 §10.4: turn-0 arrival upper bound; beyond = rejected (ns).
    // Backport fix (2026-08-16, sh_2.0测试 §5.1): default UNBOUNDED (0 = no
    // cap). The window is an explicit experiment knob only; any drop it
    // causes is counted and fail-closes the run-end completion audit.
    uint64_t request_max_arrival_ns = 0;
};

/// Parse argv for the online family. Returns false and fills `error` on any
/// contract violation; leaves `out` untouched on failure.
bool parse_online_cli(int argc, char* argv[], OnlineCliOptions& out,
                      std::string& error);

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_ONLINECLI_HH
