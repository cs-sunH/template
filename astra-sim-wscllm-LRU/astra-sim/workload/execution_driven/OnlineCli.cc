/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

OnlineCli -- execution-driven mechanism layer (wscllm phase 1).
Implementation (方案 §4 步骤 1-2 操作 5; CLI rules unit-tested in
tests/cli_online_test.cc).
*******************************************************************************/

#include "astra-sim/workload/execution_driven/OnlineCli.hh"

#include <sys/stat.h>
#include <unistd.h>

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

// Splits "--name=value" into name/value and reports via has_inline_value
// whether an inline "=" value was present (the per-option handlers own the
// missing-value error paths).
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

// Strict decimal lexicon for --idle-watchdog-s (2026-09-29): an optional
// whole part (ASCII digits) with an optional fraction, or a bare fraction;
// then an optional decimal exponent (sign allowed only after e/E). This is
// exactly the shape set strtod may honor for a plain decimal -- the
// hexadecimal-float form strtod also parses ("0x10" == 16.0, which passed
// the old whitespace/sign-only precheck) and the inf/nan words are rejected
// here before strtod can misread them as a watchdog duration.
bool is_strict_decimal(const std::string& s) {
    std::string::size_type i = 0;
    std::string::size_type digits = 0;
    if (i < s.size() && s[i] == '.') {
        ++i;  // bare-fraction form: ".5"
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
            ++i;
            ++digits;
        }
    } else {
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
            ++i;
            ++digits;
        }
        if (i < s.size() && s[i] == '.') {
            ++i;  // optional fraction: "0.5", "5."
            while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
                ++i;
                ++digits;
            }
        }
    }
    if (digits == 0) {
        return false;
    }
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
        ++i;
        if (i < s.size() && (s[i] == '+' || s[i] == '-')) {
            ++i;
        }
        std::string::size_type exp_digits = 0;
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
            ++i;
            ++exp_digits;
        }
        if (exp_digits == 0) {
            return false;  // "1e", "1e+": no exponent digits
        }
    }
    return i == s.size();
}

}  // namespace

// ---------------------------------------------------------------------------
// Exported family-grammar helpers (declared in OnlineCli.hh). They live OUT
// of the anonymous namespace because main_online.cc's argv strip must share
// this exact family boundary with parse_online_cli (M5-03, 2026-09-26: the
// shared CmdLineParser fail-closes on unknown options, so the online family
// must be stripped from argv before the shared parser re-parses it -- the
// raw handover aborted the run with "Option 'online-mode' does not exist"
// before the bridge FIFOs existed, smoke failure 2026-09-26 round 1 full).
// ---------------------------------------------------------------------------

bool is_online_family_option(const std::string& token) {
    return token.rfind("--request-", 0) == 0 || token.rfind("--bridge-", 0) == 0 ||
           token.rfind("--close-", 0) == 0 || token.rfind("--online-", 0) == 0 ||
           token.rfind("--command-", 0) == 0 || token.rfind("--sensing-", 0) == 0 ||
           // P0-2 (2026-08-31, 总文档 §4 P0-2.3): the --idle- family
           // (--idle-watchdog-s). Without this prefix the shared parser
           // would swallow a typo'd --idle-* flag (the header contract: the
           // online family is parsed fail-closed here).
           token.rfind("--idle-", 0) == 0;
}

bool online_family_option_takes_value(const std::string& name) {
    // Exact mirror of the value-taking branches in parse_online_cli below:
    //   value forms   --online-mode / --online-validate / --bridge-dir /
    //                  --request-queue-csv / --command-fifo /
    //                  --request-max-arrival-ns / --bridge-timeout-ms /
    //                  --idle-watchdog-s
    //   boolean flags --close-input / --sensing-enabled (inline "=" on a
    //                  flag is rejected by the parser, so it never consumes
    //                  a separate token either)
    return name == "--online-mode" || name == "--online-validate" ||
           name == "--bridge-dir" || name == "--request-queue-csv" ||
           name == "--command-fifo" || name == "--request-max-arrival-ns" ||
           name == "--bridge-timeout-ms" || name == "--idle-watchdog-s";
}

std::vector<char*> strip_online_family_argv(const int argc, char* argv[]) {
    std::vector<char*> out;
    if (argc <= 0) {
        return out;
    }
    out.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string token(argv[i]);
        if (!is_online_family_option(token)) {
            out.push_back(argv[i]);
            continue;
        }
        std::string name;
        std::string value;
        bool has_inline_value = false;
        split_flag(token, name, value, has_inline_value);
        // A separate-value online option consumes the NEXT token exactly the
        // way parse_online_cli does (blind argv[++i], whatever it is) -- the
        // strip must mirror that consumption so the two passes can never
        // disagree about token ownership. Inline "=" forms carry their value
        // inside the token, so nothing extra is skipped.
        if (!has_inline_value && online_family_option_takes_value(name) &&
            i + 1 < argc) {
            ++i;
        }
    }
    return out;
}

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
        if (!is_online_family_option(token)) {
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
            // F1-CPP C1-6 口径（2026-10-01 补齐）：fail-closed 而不是把
            // "--" 前缀的字面量存成值——"--bridge-dir --close-input"
            // （分离式）曾吞掉后一个 token 并静默不武装任何东西；
            // "--bridge-dir=--close-input"（内联式）曾把字面量目录名
            // 存入、在 repo 根物化 "--close-input" 目录并反转关闭语
            // 义。两种形态同一守卫。
            if (value.rfind("--", 0) == 0) {
                error = "option " + name + " requires a value (the value "
                        "token looks like another option, not a "
                        "value: " + value + ")";
                return false;
            }
            // 显式空值（"--request-queue-csv="）此前存为 ""，下游与
            // 缺席不可区分：可读性检查被跳过、expected_requests 恒 0，
            // 完成审计/静态到达门/交付门全按零分母跑（零请求 run 绿色
            // 退出）。空值是硬解析错误，与缺席不同。
            if (value.empty()) {
                error = "option " + name + " requires a non-empty value "
                        "(an explicitly empty value is not the same as "
                        "omitting the option)";
                return false;
            }
            struct stat path_stat;
            if (name == "--bridge-dir") {
                // 合法性门：存在但非目录在这里失败（而不是稍后在
                // FileDecisionBridge::ensure_bridge_dir）；不存在的路径
                // 合法——ensure_bridge_dir 负责创建（官方 runner 的
                // "C++ 先行：建 bridge 目录 + FIFO" 契约）。
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
                // FIFO 必须在解析时已存在（fixture 脚本在二进制启动前
                // mkfifo——OnlineCli.hh 契约）。读取线程旧的软
                // "cannot open" 日志会让输入门整个 run 静默敞开
                // （--idle-watchdog-s 0 下则永挂）。
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
            // for fixtures; campaigns use >= 600). The shipped default is
            // 1.0 (armed, 2026-09-05); the explicit value 0 = off restores
            // the unbounded wait_for_work() contract.
            // FP1 (2026-09-01, sync-A16 batch P; E25/E26) frozen lexicon and
            // the ONE ordering that removes the "0 = off" ambiguity,
            // tightened to the strict decimal lexicon (2026-09-29):
            //   1) token non-empty, contains no whitespace character, and
            //      its first character is neither '+' nor '-' (strtod
            //      accepts " +1"/" -1"/"\t-1" shapes that no CLI contract
            //      should honor);
            //   2) strict decimal shape (digits with optional fraction and
            //      optional exponent): rejects the hexadecimal-float form
            //      strtod would silently honor ("0x10" == 16.0 passed the
            //      old whitespace/sign-only precheck), the inf/nan words
            //      and garbage tails before strtod runs;
            //   3) errno=0 strtod, reject ERANGE (overflow AND underflow:
            //      a subnormal underflow would round to 0.0 and be
            //      misread as "off");
            //   4) reject non-finite (nan/inf degenerate the deadline) --
            //      defense in depth behind (2);
            //   5) == 0.0 -> ACCEPT, meaning OFF (E26: zero closes the
            //      watchdog only; the main loop then takes the original
            //      blocking wait_for_work(), never the checked helper);
            //   6) 0 < v < kMinIdleWatchdogSeconds -> reject ("below clock
            //      resolution": parses but could never survive the tick
            //      conversion);
            //   7) v > kMaxIdleWatchdogSeconds -> reject (platform cap).
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
            if (lex_bad) {
                error = "option --idle-watchdog-s requires a non-negative "
                        "number of seconds with no whitespace or sign "
                        "characters, got: " + value;
                return false;
            }
            if (!is_strict_decimal(value)) {
                error = "option --idle-watchdog-s requires a finite number "
                        "of seconds in strict decimal form (digits with an "
                        "optional fraction and exponent; hexadecimal like "
                        "0x10, inf and nan are rejected), got: " + value;
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
    // 互斥门 C3（2026-10-01 补齐）：--close-input 与 --command-fifo 互斥。
    // FIFO 模式下输入关闭信号属于生产者的 CloseInput/EndOfFile 命令；
    // --close-input 在启动即关门（main_online 的无 CSV 臂），FIFO 通道
    // 一开始就是死的：后续每个 Submit/CloseInput 都被关掉的门拒绝，
    // run 以看似健康的零请求形态收场（完成审计分母 0 被跳过）。
    // 官方 runner 从不组合二者；本仓 --close-input 为裸旗标词法
    // （带值形态一律硬错），不传该旗标即关闭态、配 FIFO 合法。
    if (close_input && !command_fifo.empty()) {
        error =
            "--close-input and --command-fifo are mutually exclusive: "
            "--close-input would close the input gate at startup, making "
            "the FIFO producer channel dead (a zero-request run that exits "
            "green). Let the producer signal CloseInput/EndOfFile over the "
            "FIFO instead";
        return false;
    }
    // 互斥门 3.16（2026-10-01 补齐）：--request-queue-csv 与
    // --command-fifo 亦互斥。组合能合法解析，但 CSV 臂在 ingress 上装
    // 到达钩子、其 queue-index 域是 CSV 数据行序——FIFO 信封的
    // queue_index 停在缺省 -1，第一个 FIFO Submit 会死在
    // notify_consumed(-1) 深处、报错指向错误子系统。在两个旗标都可见
    // 的解析边界拒绝。
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
