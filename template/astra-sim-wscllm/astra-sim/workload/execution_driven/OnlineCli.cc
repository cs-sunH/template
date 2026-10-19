/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

OnlineCli -- execution-driven mechanism layer (wscllm phase 1).
Implementation (方案 §4 步骤 1-2 操作 5; CLI rules unit-tested in
tests/cli_online_test.cc).
*******************************************************************************/

#include "astra-sim/workload/execution_driven/OnlineCli.hh"

#include <unistd.h>

#include <sys/stat.h>

#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

namespace AstraSim {
namespace ExecutionDriven {

namespace {

// (The online-family prefix lexicon lives in OnlineCli.hh's single
// metadata table -- network-F7 fix; is_online_family_token below is the
// shared query. The former file-local mirror is gone.)

// FP1 (2026-09-01, sync-A16 batch P; contract E25): the frozen unsigned
// integer lexicon -- non-empty and every character an ASCII '0'..'9'.
// This replaces the old first-character '-' rejection: strtoull skips
// leading whitespace, so " -1" / "\t-1" passed the old check, parsed as a
// negated wrap-around ULLONG_MAX, and set no ERANGE (the 64-bit range
// check could not catch it); "+1" was silently accepted as 1. A pure
// digit lexicon rejects all three shapes before strtoull runs.
bool is_ascii_digits(const std::string& s) {
    if (s.empty()) {
        return false;
    }
    for (const char c : s) {
        if (c < '0' || c > '9') {
            return false;
        }
    }
    return true;
}

// Splits "name=value": has_inline_value reports whether an "=" value was
// attached (a bare flag leaves value empty).
void split_flag(const std::string& token, std::string& name,
                std::string& value, bool& has_inline_value) {
    const auto eq = token.find('=');
    if (eq == std::string::npos) {
        name = token;
        has_inline_value = false;
        value.clear();
        return;
    }
    name = token.substr(0, eq);
    value = token.substr(eq + 1);
    has_inline_value = true;
}

}  // namespace

bool parse_online_cli(const int argc, char* argv[], OnlineCliOptions& out,
                      std::string& error) {
    std::string mode;
    bool close_input = false;
    bool sensing_enabled = false;
    std::string bridge_dir;
    int bridge_timeout_ms = 0;
    // P0-2 (2026-08-31, 总文档 §4 P0-2.3): wall-clock parking watchdog
    // seconds. Default 1.0 since 2026-09-05 (ships armed; explicit 0 = off
    // restores the unbounded wait_for_work contract for fixtures/IDLE
    // runs). Must stay in sync with OnlineCliOptions::idle_watchdog_s --
    // this local is written back to `out` unconditionally, so it IS the
    // binary's behavioral default.
    double idle_watchdog_s = 1.0;
    std::string request_queue_csv;
    std::string command_fifo;
    // Backport fix (2026-08-16, sh_2.0测试 §5.1): default UNBOUNDED (0 = no
    // arrival-window cap). The previous 30e9 default burned the 30s
    // acceptance-input window into the code and silently dropped over-window
    // requests; the window survives only as an explicit experiment knob
    // (--request-max-arrival-ns), and any drop it causes is fail-closed at
    // the run-end completion audit (see main_online.cc).
    uint64_t request_max_arrival_ns = 0;
    // C1 validate switch (2026-08-28), B.2 cleanup (2026-09-05): strict
    // <0|1> enum -- 1 full / 0 off; the N >= 2 sample-every-Nth tier is
    // removed. Frozen CLI default 1 (fail-closed, matches the pre-C1
    // behavior for bare invocations); the official runner overrides with
    // "${SH_ONLINE_VALIDATE:-0}".
    int online_validate = 1;

    for (int i = 1; i < argc; ++i) {
        const std::string token(argv[i]);
        if (token.rfind("--", 0) != 0) {
            // A non-option token here is the separated VALUE of a shared
            // (non-online-family) option, e.g. the path after
            // --workload-configuration: online-family options consume their
            // own values in the arms below (P2-4/P2-5, 2026-09-28), so this
            // token must be left untouched for CmdLineParser, which owns
            // the shared options and consumes the value with them. A truly
            // stray positional rides along and is dropped into cxxopts'
            // unmatched list (verified: with no parse_positional() target
            // defined, consume_positional() returns false and the token is
            // silently discarded) -- this loop has no positional grammar of
            // its own and must not guess.
            continue;
        }
        if (!is_online_family_token(token)) {
            continue;  // shared static options: left to CmdLineParser
        }
        std::string name;
        std::string value;
        bool has_inline_value = false;
        split_flag(token, name, value, has_inline_value);

        if (name == "--online-mode") {
            if (!has_inline_value) {
                if (i + 1 >= argc) {
                    error = "option --online-mode requires a value";
                    return false;
                }
                value = argv[++i];
            }
            // Path-2 removal (2026-08-18): replay mode was deleted with the
            // replay route; strategy is the only legal mode token.
            if (value != "strategy") {
                error = "unknown --online-mode value: " + value +
                        " (expected \"strategy\"; replay was removed with "
                        "the replay route on 2026-08-18)";
                return false;
            }
            mode = value;
        } else if (name == "--online-validate") {
            // C1 validate switch (2026-08-28), B.2 cleanup (2026-09-05):
            // strict <0|1> enum (same shape as the former --online-node-gc
            // enum, whose CLI arm the B.3 cleanup (2026-09-05) removed);
            // the N >= 2
            // sample-every-Nth-batch tier is removed. 1 = full validation
            // (pre-C1 behavior), 0 = production skip (commit() keeps its
            // mandatory liveness preflight). Any other token -- including
            // the former sampling N >= 2 and the wrap-around shapes the old
            // numeric lexicon guarded against (4294967296 would have
            // silently DISABLED validation, fail-open) -- is a hard parse
            // error, so a stale script can never silently change validation
            // frequency.
            if (!has_inline_value) {
                if (i + 1 >= argc) {
                    error = "option --online-validate requires a value";
                    return false;
                }
                value = argv[++i];
            }
            if (value == "0") {
                online_validate = 0;
            } else if (value == "1") {
                online_validate = 1;
            } else {
                error = "unknown --online-validate value: " + value +
                        " (expected \"0\" or \"1\"; the N >= 2 sampled "
                        "tier was removed on 2026-09-05)";
                return false;
            }
        } else if (name == "--close-input" || name == "--sensing-enabled") {
            // P2-5 fix (2026-09-28): unified boolean-flag semantics (see
            // online_boolean_literal in the header). The pre-fix parser
            // rejected the inline "=..." form while silently IGNORING a
            // separated value token (it is not "--"-prefixed, so the loop
            // skipped it as a positional): "--close-input false" armed the
            // close and "--sensing-enabled false" enabled sensing -- both
            // intents silently inverted. Every form now parses explicitly:
            //   bare flag                  -> true
            //   --flag=true|1 / =false|0   -> inline value
            //   --flag true|1 / false|0    -> separated value (consumed)
            //   --flag <anything else>     -> hard parse error
            // A "--"-prefixed next token is NOT a value: the flag stands
            // alone (true) and the option after it is parsed normally.
            bool flag_true = true;
            if (has_inline_value) {
                const int lit = online_boolean_literal(value);
                if (lit < 0) {
                    error = "flag " + name + " requires a boolean value "
                            "(true/1/false/0), got: " + value;
                    return false;
                }
                flag_true = lit == 1;
            } else if (i + 1 < argc) {
                const std::string next(argv[i + 1]);
                if (next.rfind("--", 0) != 0) {
                    const int lit = online_boolean_literal(next);
                    if (lit < 0) {
                        error = "flag " + name + " requires a boolean "
                                "value (true/1/false/0), got: " + next;
                        return false;
                    }
                    ++i;  // consume the explicit separated value
                    flag_true = lit == 1;
                }
            }
            if (name == "--close-input") {
                close_input = flag_true;
            } else {
                sensing_enabled = flag_true;
            }
        } else if (name == "--bridge-dir" || name == "--request-queue-csv" ||
                   name == "--command-fifo") {
            if (!has_inline_value) {
                if (i + 1 >= argc) {
                    error = "option " + name + " requires a value";
                    return false;
                }
                value = argv[++i];
            }
            // P2-4 fix (2026-09-28), F1-CPP C1-6 fix (2026-09-29): fail
            // closed instead of storing a "--"-prefixed literal as the
            // value. The guard now covers BOTH value forms: it originally
            // sat inside the !has_inline_value branch, so the inline shape
            // "--bridge-dir=--close-input" slipped through, stored the
            // literal directory name and INVERTED the close semantics
            // exactly like the separated shape -- under the official
            // runner (which cd's to the repo root first) ensure_bridge_dir
            // materialized a "--close-input" directory in the repo root
            // and the runner's no-timeout FIFO wait loop wedged both
            // sides. "--bridge-dir --close-input" (separated) used to
            // swallow the token and silently arm nothing -- same poison,
            // same diagnosis.
            if (value.rfind("--", 0) == 0) {
                error = "option " + name + " requires a value (the value "
                        "token looks like another option, not a "
                        "value: " + value + ")";
                return false;
            }
            // P2-6 fix (2026-09-28): an explicitly given but EMPTY value
            // ("--request-queue-csv=") used to be stored as "" and
            // indistinguishable from the option being absent downstream:
            // the readability check was skipped, expected_requests stayed 0
            // and the completion audit / static-arrival gate / delivery
            // gate all ran against a zero denominator (a zero-request run
            // exits green). An empty value is now a hard parse error,
            // distinct from omitting the option.
            if (value.empty()) {
                error = "option " + name + " requires a non-empty value "
                        "(an explicitly empty value is not the same as "
                        "omitting the option)";
                return false;
            }
            struct stat path_stat;
            if (name == "--bridge-dir") {
                // P2-4 fix: legality gate. A path that exists but is not a
                // directory fails here instead of later in
                // FileDecisionBridge::ensure_bridge_dir. A NON-existent
                // path stays legal: ensure_bridge_dir creates the directory
                // (the official runner's documented "C++ first: creates
                // the bridge dir + FIFOs" contract).
                if (::stat(value.c_str(), &path_stat) == 0 &&
                    !S_ISDIR(path_stat.st_mode)) {
                    error = "option --bridge-dir exists but is not a "
                            "directory: " + value;
                    return false;
                }
                bridge_dir = value;
            } else if (name == "--request-queue-csv") {
                request_queue_csv = value;
            } else {
                // P2-4 fix: the FIFO must exist at parse time (the fixture
                // script mkfifs it before the binary starts -- OnlineCli.hh
                // contract). The reader thread's old soft "cannot open"
                // log left the input gate silently open for the whole run.
                if (::stat(value.c_str(), &path_stat) != 0 ||
                    !S_ISFIFO(path_stat.st_mode)) {
                    error = "option --command-fifo must name an existing "
                            "FIFO (mkfifo it before the binary starts): " +
                            value;
                    return false;
                }
                command_fifo = value;
            }
        } else if (name == "--request-max-arrival-ns" ||
                   name == "--bridge-timeout-ms") {
            // WindowedTraceReader knob (turn-0 arrival upper bound) + the
            // bridge poll watchdog.
            // Non-negative integers. FP1 (2026-09-01, sync-A16 batch P;
            // E25): frozen lexicon -- pure ASCII digits (rejects " -1",
            // "\t-1", "+1" before strtoull can wrap them), errno=0 +
            // ERANGE rejection, endptr-at-end, then the per-option target
            // type range; no partial writes on failure.
            if (!has_inline_value) {
                if (i + 1 >= argc) {
                    error = "option " + name + " requires a value";
                    return false;
                }
                value = argv[++i];
            }
            if (!is_ascii_digits(value)) {
                error = "option " + name + " requires a non-negative "
                        "integer (ASCII digits only), got: " + value;
                return false;
            }
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = std::strtoull(
                value.c_str(), &end, 10);
            if (errno == ERANGE || end == value.c_str() || *end != '\0') {
                error = "option " + name + " exceeds the unsigned integer "
                        "range, got: " + value;
                return false;
            }
            if (name == "--bridge-timeout-ms") {
                if (parsed > 2147483647ULL) {
                    error = "option --bridge-timeout-ms exceeds int range, "
                            "got: " + value;
                    return false;
                }
                bridge_timeout_ms = static_cast<int>(parsed);
            } else {
                request_max_arrival_ns = static_cast<uint64_t>(parsed);
            }
        } else if (name == "--idle-watchdog-s") {
            // P0-2 (2026-08-31, 总文档 §4 P0-2.3): wall-clock event-loop
            // parking watchdog. Double seconds (sub-second armings are legal
            // for fixtures; campaigns use >= 600), 0 = off (the default arms
            // at 1.0 s since 2026-09-05 -- ships armed; OnlineCli.hh).
            // FP1 (2026-09-01, sync-A16 batch P; E25/E26) frozen lexicon and
            // the ONE ordering that removes the "0 = off" ambiguity:
            //   1) token non-empty, contains no whitespace character, and
            //      its first character is neither '+' nor '-' (strtod
            //      accepts " +1"/" -1"/"\t-1" shapes that no CLI contract
            //      should honor);
            //   2) errno=0 strtod, reject ERANGE (overflow AND underflow:
            //      a subnormal underflow would round to 0.0 and be
            //      misread as "off");
            //   3) reject non-finite (nan/inf degenerate the deadline);
            //   4) == 0.0 -> ACCEPT, meaning OFF (E26: zero closes the
            //      watchdog only; the main loop then takes the original
            //      blocking wait_for_work(), never the checked helper);
            //   5) 0 < v < kMinIdleWatchdogSeconds -> reject ("below clock
            //      resolution": parses but could never survive the tick
            //      conversion);
            //   6) v > kMaxIdleWatchdogSeconds -> reject (platform cap).
            // The bounds themselves (1e-9 / 1e9) are accepted.
            if (!has_inline_value) {
                if (i + 1 >= argc) {
                    error = "option --idle-watchdog-s requires a value";
                    return false;
                }
                value = argv[++i];
            }
            bool lex_bad = value.empty();
            bool exponent_marker_just_seen = false;
            for (const char c : value) {
                if (std::isspace(static_cast<unsigned char>(c))) {
                    lex_bad = true;
                    break;
                }
                // Hex-float lex gap (2026-10-01 fix): std::strtod accepts
                // "0x10"/"0x1p4" (hex / hex-float literals), which no
                // documented value form of this option uses. Restrict the
                // token to the plain decimal grammar [0-9] '.' 'e'/'E',
                // allowing ONE sign ONLY immediately after an exponent
                // marker (so "1e-320" stays a legal subnormal literal)
                // -- signs anywhere else, and every hex/x/X/nan/inf token,
                // die here with the lex diagnostic.
                if (c >= '0' && c <= '9') {
                    exponent_marker_just_seen = false;
                    continue;
                }
                if (c == '.') {
                    exponent_marker_just_seen = false;
                    continue;
                }
                if (c == 'e' || c == 'E') {
                    exponent_marker_just_seen = true;
                    continue;
                }
                if ((c == '+' || c == '-') &&
                    exponent_marker_just_seen) {
                    exponent_marker_just_seen = false;
                    continue;
                }
                lex_bad = true;
                break;
            }
            if (!value.empty() && (value[0] == '+' || value[0] == '-')) {
                lex_bad = true;
            }
            if (lex_bad) {
                error = "option --idle-watchdog-s requires a plain "
                        "non-negative decimal number of seconds (digits, "
                        "dot, exponent only; no whitespace, signs, or hex), "
                        "got: " + value;
                return false;
            }
            char* end = nullptr;
            errno = 0;
            const double parsed_s =
                std::strtod(value.c_str(), &end);
            if (errno == ERANGE || end == value.c_str() || *end != '\0' ||
                !std::isfinite(parsed_s)) {
                error = "option --idle-watchdog-s requires a finite number "
                        "of seconds within the accepted range, got: " +
                        value;
                return false;
            }
            if (parsed_s == 0.0) {
                // E26: 0 = off -- accepted, no bound applies.
                idle_watchdog_s = parsed_s;
            } else if (parsed_s < kMinIdleWatchdogSeconds) {
                error = "option --idle-watchdog-s is below the clock "
                        "resolution (minimum " +
                        std::to_string(kMinIdleWatchdogSeconds) +
                        " s), got: " + value;
                return false;
            } else if (parsed_s > kMaxIdleWatchdogSeconds) {
                error = "option --idle-watchdog-s exceeds the accepted "
                        "upper bound (maximum " +
                        std::to_string(kMaxIdleWatchdogSeconds) +
                        " s), got: " + value;
                return false;
            } else {
                idle_watchdog_s = parsed_s;
            }
        } else {
            error = "unknown online-family option: " + name;
            return false;
        }
    }

    if (mode.empty()) {
        error = "missing required option --online-mode";
        return false;
    }
    // C3 fix (2026-09-28): --close-input and --command-fifo are mutually
    // exclusive. With a FIFO the input-close signaling belongs to the
    // producer's CloseInput/EndOfFile commands; --close-input closes the
    // gate immediately at startup (no-CSV arm of the close rule in
    // main_online), so the FIFO channel is dead on arrival: every later
    // Submit/CloseInput command is rejected by the closed gate and the run
    // finishes as a seemingly healthy ZERO-request run (no CSV -> the
    // completion audit's denominator is 0 and is skipped). No official
    // runner combines the two (the CSV runners pair --close-input with
    // --request-queue-csv; the FIFO fixtures never pass --close-input).
    // --close-input=false with a FIFO stays legal: the flag is explicitly
    // off.
    if (close_input && !command_fifo.empty()) {
        error =
            "--close-input and --command-fifo are mutually exclusive: "
            "--close-input would close the input gate at startup, making "
            "the FIFO producer channel dead (a zero-request run that exits "
            "green). Let the producer signal CloseInput/EndOfFile over the "
            "FIFO instead";
        return false;
    }
    // 3.16 fix (2026-09-29): --request-queue-csv and --command-fifo are
    // mutually exclusive too. The combination parses legally (only the
    // close-input pair was excluded above), but the CSV arm installs the
    // arrival hook on the ingress and its queue-index domain is the CSV
    // data-row order -- a FIFO envelope's queue_index stays at the default
    // -1, so the first FIFO Submit died in notify_consumed(-1) deep inside
    // the reader's accounting with a message about the wrong subsystem.
    // Reject the combination at the parse boundary where both flags are
    // visible.
    if (!request_queue_csv.empty() && !command_fifo.empty()) {
        error =
            "--request-queue-csv and --command-fifo are mutually exclusive: "
            "the CSV path owns the ingress arrival hooks and the queue-index "
            "domain, while FIFO Submit envelopes arrive with no CSV queue "
            "index (the combination aborted on the first FIFO Submit). Pick "
            "one input channel";
        return false;
    }
    if (!request_queue_csv.empty()) {
        if (access(request_queue_csv.c_str(), R_OK) != 0) {
            error = "--request-queue-csv is not readable: " +
                    request_queue_csv;
            return false;
        }
    }

    out.mode = mode;
    out.close_input = close_input;
    out.bridge_dir = bridge_dir;
    out.bridge_timeout_ms = bridge_timeout_ms;
    out.idle_watchdog_s = idle_watchdog_s;
    out.request_queue_csv = request_queue_csv;
    out.command_fifo = command_fifo;
    out.sensing_enabled = sensing_enabled;
    out.request_max_arrival_ns = request_max_arrival_ns;
    out.online_validate = online_validate;
    return true;
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
