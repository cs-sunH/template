/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

OnlineCli -- execution-driven mechanism layer (wscllm phase 1).
Implementation (方案 §4 步骤 1-2 操作 5; CLI rules unit-tested in
tests/cli_online_test.cc).
*******************************************************************************/

#include "astra-sim/workload/execution_driven/OnlineCli.hh"

#include <unistd.h>

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
           token.rfind("--command-", 0) == 0 || token.rfind("--sensing-", 0) == 0;
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

}  // namespace

bool parse_online_cli(const int argc, char* argv[], OnlineCliOptions& out,
                      std::string& error) {
    std::string mode;
    bool close_input = false;
    bool sensing_enabled = false;
    std::string bridge_dir;
    std::string request_queue_csv;
    std::string command_fifo;
    size_t request_window_rows = 128;
    // Backport fix 2026-08-16 (对比报告 §5.1): the default arrival window is
    // UNBOUNDED. The old default (30,000,000,000 ns) was the 30s acceptance
    // input's window assumption frozen into the mechanism layer -- any longer
    // input had its later turn-0 rows silently rejected by the windowed
    // reader (12.5% measured on a 3-minute input) while the run still
    // reported PASS. The window bound stays available as an EXPLICIT
    // parameter; when set, out-of-window rows are counted and fail the run
    // (main_online run-end audit), never silently dropped.
    uint64_t request_max_arrival_ns = 0;

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
            if (value != "replay" && value != "strategy") {
                error = "unknown --online-mode value: " + value +
                        " (expected \"replay\" or \"strategy\")";
                return false;
            }
            mode = value;
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
            if (name == "--bridge-dir") {
                bridge_dir = value;
            } else if (name == "--request-queue-csv") {
                request_queue_csv = value;
            } else {
                command_fifo = value;
            }
        } else if (name == "--request-window-rows" ||
                   name == "--request-max-arrival-ns") {
            // Phase 7 §10.4: WindowedTraceReader knobs (window high water /
            // turn-0 arrival upper bound). Non-negative integers.
            if (!has_inline_value) {
                if (i + 1 >= argc) {
                    error = "option " + name + " requires a value";
                    return false;
                }
                value = argv[++i];
            }
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(
                value.c_str(), &end, 10);
            if (value.empty() || value[0] == '-' || end == value.c_str() ||
                *end != '\0') {
                error = "option " + name + " requires a non-negative "
                        "integer, got: " + value;
                return false;
            }
            if (name == "--request-window-rows") {
                request_window_rows = static_cast<size_t>(parsed);
            } else {
                request_max_arrival_ns = static_cast<uint64_t>(parsed);
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
    out.request_queue_csv = request_queue_csv;
    out.command_fifo = command_fifo;
    out.sensing_enabled = sensing_enabled;
    out.request_window_rows = request_window_rows;
    out.request_max_arrival_ns = request_max_arrival_ns;
    return true;
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
