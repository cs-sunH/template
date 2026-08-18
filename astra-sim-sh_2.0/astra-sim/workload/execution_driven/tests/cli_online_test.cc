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

Build (from template/astra-sim-wscllm):
  g++ -std=c++17 -I . astra-sim/workload/execution_driven/tests/cli_online_test.cc \
      astra-sim/workload/execution_driven/OnlineCli.cc \
      -o /tmp/cli_online_test && /tmp/cli_online_test
*******************************************************************************/

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

    // R5: --close-input flag
    assert(parse_ok({"--online-mode", "strategy", "--close-input"}, out));
    assert(out.close_input);
    // R5: value form rejected
    assert(parse_error({"--online-mode", "strategy", "--close-input=true"})
               .find("no value") != std::string::npos);

    // R6: --bridge-dir parsed and stored
    assert(parse_ok({"--online-mode", "strategy", "--bridge-dir=/tmp/bridge"},
                    out));
    assert(out.bridge_dir == "/tmp/bridge");
    // R6: space-separated form
    assert(parse_ok({"--online-mode", "strategy", "--bridge-dir", "/tmp/bridge"},
                    out));
    assert(out.bridge_dir == "/tmp/bridge");

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

    // R9: --command-fifo parsed and stored (inline and separate-value forms)
    assert(parse_ok({"--online-mode", "strategy",
                     "--command-fifo=/tmp/fixture_cmd.fifo"}, out));
    assert(out.command_fifo == "/tmp/fixture_cmd.fifo");
    assert(parse_ok({"--online-mode", "strategy", "--command-fifo",
                     "/tmp/fixture_cmd.fifo"}, out));
    assert(out.command_fifo == "/tmp/fixture_cmd.fifo");
    // R7: --command-* typos are hard errors
    assert(parse_error({"--online-mode", "strategy", "--command-fifoX=/x"})
               .find("unknown online-family") != std::string::npos);

    // missing --online-mode even with other online flags
    assert(parse_error({"--request-queue-csv=/proc/self/exe", "--close-input"})
               .find("--online-mode") != std::string::npos);

    // R11: phase-7 §10.4 --request-window-rows / --request-max-arrival-ns
    // (defaults: 128 rows / unbounded arrival window -- backport fix
    // 2026-08-16 对比报告 §5.1; the old 30e9 default silently rejected
    // later turn-0 rows of longer inputs)
    assert(parse_ok({"--online-mode", "strategy"}, out));
    assert(out.request_window_rows == 128);
    assert(out.request_max_arrival_ns == 0);
    assert(parse_ok({"--online-mode", "strategy",
                     "--request-window-rows=64"}, out));
    assert(out.request_window_rows == 64);
    assert(parse_ok({"--online-mode", "strategy", "--request-window-rows",
                     "0"}, out));
    assert(out.request_window_rows == 0);  // unbounded control arm
    assert(parse_ok({"--online-mode", "strategy",
                     "--request-max-arrival-ns=12345"}, out));
    assert(out.request_max_arrival_ns == 12345);
    // R11: non-integer / negative values are hard errors
    assert(parse_error({"--online-mode", "strategy",
                        "--request-window-rows=abc"})
               .find("non-negative integer") != std::string::npos);
    assert(parse_error({"--online-mode", "strategy",
                        "--request-max-arrival-ns=-5"})
               .find("non-negative integer") != std::string::npos);
    // R11: missing value forms rejected
    assert(parse_error({"--online-mode", "strategy",
                        "--request-window-rows"})
               .find("requires a value") != std::string::npos);
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
    // R10: value form rejected
    assert(parse_error({"--online-mode", "strategy", "--sensing-enabled=true"})
               .find("takes no value") != std::string::npos);
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

    std::printf("[cli] ALL PASS: R1-R11 online CLI contract verified "
                "(incl. phase-7 §10.4 window knobs)\n");
    return 0;
}
