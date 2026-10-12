/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

OnlineCli -- execution-driven mechanism layer (wscllm phase 1).
Implementation (方案 §4 步骤 1-2 操作 5).
*******************************************************************************/

#include "astra-sim/workload/execution_driven/OnlineCli.hh"

#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

namespace AstraSim {
namespace ExecutionDriven {

namespace {

bool is_online_family(const std::string& token) {
    return token.rfind("--request-", 0) == 0 || token.rfind("--bridge-", 0) == 0 ||
           token.rfind("--close-", 0) == 0 || token.rfind("--online-", 0) == 0 ||
           token.rfind("--command-", 0) == 0 || token.rfind("--sensing-", 0) == 0 ||
           // P0-2 (2026-08-31, 总文档 §4 P0-2.3): the --idle- family
           // (--idle-watchdog-s). Without this prefix the shared parser
           // would silently swallow a typo'd --idle-* flag (the header
           // contract: the online family is parsed fail-closed here).
           token.rfind("--idle-", 0) == 0;
}

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

// Returns the inline value after "=", or false when the value is missing.
bool split_flag(const std::string& token, std::string& name,
                std::string& value, bool& has_inline_value) {
    const auto eq = token.find('=');
    if (eq == std::string::npos) {
        name = token;
        has_inline_value = false;
        value.clear();
        return true;
    }
    name = token.substr(0, eq);
    value = token.substr(eq + 1);
    has_inline_value = true;
    return true;
}

// (2026-10-01) Render a watchdog bound for diagnostics: std::to_string
// (double) prints %f with 6 decimals, so the 1e-9 lower bound rendered as
// "0.000000" and contradicted the rejection it justified. %g keeps the
// exponent visible for sub-microsecond and gigasecond bounds.
std::string render_seconds(const double seconds) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.3g", seconds);
    return std::string(buffer);
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
            continue;  // positional / value token; handled by the consumer
        }
        if (!is_online_family(token)) {
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
        } else if (name == "--close-input") {
            if (has_inline_value) {
                error = "flag --close-input takes no value";
                return false;
            }
            close_input = true;
        } else if (name == "--sensing-enabled") {
            // Phase-3 perception feature flag (default off until phase 6).
            if (has_inline_value) {
                error = "flag --sensing-enabled takes no value";
                return false;
            }
            sensing_enabled = true;
        } else if (name == "--bridge-dir" || name == "--request-queue-csv" ||
                   name == "--command-fifo") {
            if (!has_inline_value) {
                if (i + 1 >= argc) {
                    error = "option " + name + " requires a value";
                    return false;
                }
                value = argv[++i];
            }
            // Empty-value rejection (2026-09-30, ported from the joint repo
            // OnlineCli.cc parse_online_cli): an explicitly empty inline
            // value ("--request-queue-csv=") used to be stored as "" and
            // downgraded to "absent" -- a silent zero-request IDLE run. All
            // three path options reject the empty spelling here.
            // (2026-10-01) Flag-like value rejection: "--bridge-dir
            // --close-input" (a dropped value) used to consume the next
            // flag as a path, mkdir a directory literally named
            // "--close-input", and park the bridge open retry forever --
            // reject values that themselves look like an option token.
            if (value.empty()) {
                error = "option " + name + " requires a non-empty value";
                return false;
            }
            if (value.rfind("--", 0) == 0) {
                error = "option " + name + " requires a value, got the "
                        "flag-like token " + value +
                        " (a missing value must not silently swallow the "
                        "next option)";
                return false;
            }
            if (name == "--bridge-dir") {
                bridge_dir = value;
            } else if (name == "--request-queue-csv") {
                request_queue_csv = value;
            } else {
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
            // (2026-10-01) Flag-like value rejection, same rationale as the
            // path options above: a dropped value must not silently consume
            // the following option token. NOTE: on these digit-lexicon
            // options this arm is unreachable defense ('-' fails
            // is_ascii_digits first) -- kept for symmetry with the path
            // options and future lexicon changes.
            if (value.rfind("--", 0) == 0) {
                error = "option " + name + " requires a value, got the "
                        "flag-like token " + value;
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
            // for fixtures; campaigns use >= 600), 0 = off (the default has
            // been 1.0 = armed since 2026-09-05, not 0).
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
            for (const char c : value) {
                if (std::isspace(static_cast<unsigned char>(c))) {
                    lex_bad = true;
                    break;
                }
            }
            if (!value.empty() && (value[0] == '+' || value[0] == '-')) {
                lex_bad = true;
            }
            // (2026-10-01) Decimal-only lexicon: strtod also accepts hex
            // floats ("0x10" -> 16.0) and bare-dot forms; the contract's
            // stated grammar is decimal seconds ("1e+3"/"1.5" style).
            // Restrict the first character to a digit or '.', and reject
            // the 0x prefix explicitly (".5"/"1." stay legal).
            if (!value.empty() &&
                !(std::isdigit(static_cast<unsigned char>(value[0])) ||
                  value[0] == '.')) {
                lex_bad = true;
            }
            if (value.size() >= 2 && value[0] == '0' &&
                (value[1] == 'x' || value[1] == 'X')) {
                lex_bad = true;
            }
            if (lex_bad) {
                error = "option --idle-watchdog-s requires a non-negative "
                        "decimal number of seconds with no whitespace "
                        "anywhere and no sign character at the leading "
                        "position (internal signs as in 1e+3 are legal), "
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
                        render_seconds(kMinIdleWatchdogSeconds) +
                        " s), got: " + value;
                return false;
            } else if (parsed_s > kMaxIdleWatchdogSeconds) {
                error = "option --idle-watchdog-s exceeds the accepted "
                        "upper bound (maximum " +
                        render_seconds(kMaxIdleWatchdogSeconds) +
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
