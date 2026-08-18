/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

bridge_loopback_fixture.cc -- phase-1 step 1-7 FileDecisionBridge loopback.

Scenarios (方案 §4 步骤 1-7 操作 4):
  A. Round-trip (main process): two StateDeltas covering all four phase-1
     reasons (ARRIVAL / PREFILL_DRAIN / DECODE_COMPLETION /
     REQUEST_COMPLETE) are delivered through a real FileDecisionBridge to
     online/verify/bridge_echo.py; the echo reflects the full request back
     into response["echo_of_request"], so the fixture asserts field-level
     round-trip fidelity (echo == build_request_json(delta)) plus the
     fixed nodes/parent_edges/watches/assignments/kv_actions arrays and
     batch_id/source_delivery_sequence. Then commit acks are sent twice for
     the same seq (idempotency: Python must ignore the duplicate) and the
     echo's receipt file must contain exactly one line per seq. At scope
     end the bridge closes req_notify -> Python sees EOF -> exits 0.
  B. Python crash detection (forked child): the child delivers through a
     bridge whose Python side reads the request and exits WITHOUT writing a
     response; the resp_notify read gets EOF and the child must abort
     (SIGABRT) with the "Python side crashed" message.
  C. Timeout (forked child): the child uses timeout_ms=1000 against a
     Python side that never opens resp_notify; the poll must time out and
     the child must abort (SIGABRT) with the timeout message.
  D. Genuine peer death under the long-lived protocol (defect-B fix,
     2026-08-16): Python completes exchange 1 then SIGKILLs itself; the
     C++ lifetime read end must abort with "long-lived resp_notify write
     end closed" (unambiguous death -- the same read()==0 that the old
     per-exchange protocol also produced for a LIVE peer, F1's false kill).
  E. 1:1 response-byte invariant (defect-B fix guard): a Python writing
     two bytes for one delivery must trip "more than one response byte in
     flight" (backpressure contract) instead of shifting the exchanges.

Build: the CMake target AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest
(build with bash sh_test_mesh/run_scripts/build_analytical_aware.sh).
Run (from template/astra-sim-wscllm):
  build/astra_analytical/build_congestion_aware/bin/\
      AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest \
      sh_test_mesh/workload/llama2_7b_inference/online/verify/bridge_echo.py
Exit code 0 on ALL PASS (requires python3 on PATH).
*******************************************************************************/

#include <signal.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <json/json.hpp>

#include "astra-sim/workload/execution_driven/DecisionBridge.hh"

using namespace AstraSim::ExecutionDriven;

namespace {

bool g_ok = true;

void expect(bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "[bridge_loopback_fixture] FAIL: %s\n", what);
        g_ok = false;
    }
}

std::string make_temp_root() {
    const std::string root =
        "/tmp/wscllm_bridge_loopback_" + std::to_string(::getpid());
    if (::mkdir(root.c_str(), 0755) != 0) {
        std::perror("mkdir");
        std::exit(1);
    }
    return root;
}

void rm_rf(const std::string& path) {
    const std::string cmd = "rm -rf '" + path + "'";
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::fprintf(stderr, "[bridge_loopback_fixture] rm_rf(%s) rc=%d\n",
                     path.c_str(), rc);
    }
}

pid_t spawn_python3(const std::vector<std::string>& args) {
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>("python3"));
    for (const auto& arg : args) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr);
    const pid_t pid = ::fork();
    if (pid == 0) {
        ::execvp("python3", argv.data());
        _exit(127);  // exec failed
    }
    if (pid < 0) {
        std::perror("fork");
        std::exit(1);
    }
    return pid;
}

nlohmann::json read_json_file_for_test(const std::string& path) {
    std::ifstream in(path);
    nlohmann::json value;
    in >> value;
    return value;
}

bool file_exists(const std::string& path) {
    std::ifstream in(path);
    return in.good();
}

// ---------------------------------------------------------------- Part A --
// Round-trip + ack idempotency + clean Python exit.
void round_trip(const std::string& echo_script) {
    const std::string root = make_temp_root();
    const std::string bridge_dir = root + "/bridge";
    const std::string ack_receipt = root + "/ack_receipt";
    FileDecisionBridge::ensure_bridge_dir(bridge_dir);

    const pid_t py = spawn_python3({echo_script, bridge_dir, ack_receipt});

    {
        FileDecisionBridge bridge(bridge_dir);  // timeout 0 (phase-1 default)

        // --- delta 1: ARRIVAL + PREFILL_DRAIN ---
        DecisionEvent arrival_ev;
        arrival_ev.reason = DecisionReason::ARRIVAL;
        arrival_ev.request_id = "s0_r0";
        arrival_ev.payload.session_id = "s0";
        arrival_ev.payload.turn_index = 0;
        arrival_ev.payload.prefill_length = 15659;
        arrival_ev.payload.decode_length = 202;
        arrival_ev.payload.inter_request_interval_ns = 0;
        arrival_ev.payload.arrival_world_ns = 94835000;

        DecisionEvent prefill_ev;
        prefill_ev.reason = DecisionReason::PREFILL_DRAIN;
        prefill_ev.request_id = "s0_r0";
        prefill_ev.stage = "prefill";
        prefill_ev.generation = 0;
        prefill_ev.payload.watch_member_count = 5;

        const StateDelta delta1 =
            build_state_delta({arrival_ev, prefill_ev}, 94835000, 1);

        const GraphBatch batch1 = bridge.deliver_and_receive(delta1);

        // request JSON on disk must be exactly the protocol serializer's
        // output (atomic publish means the file is complete at notify time)
        const nlohmann::json request1_on_disk =
            read_json_file_for_test(bridge_dir + "/request_1.json");
        expect(request1_on_disk == build_request_json(delta1),
               "A: request_1.json equals build_request_json(delta1)");
        expect(request1_on_disk["reasons"] ==
                   nlohmann::json::parse(R"(["ARRIVAL","PREFILL_DRAIN"])"),
               "A: reasons[] in epoch order");
        expect(request1_on_disk["arrivals"][0]["request_id"] == "s0_r0" &&
                   request1_on_disk["arrivals"][0]["arrival_world_ns"] == 94835000 &&
                   request1_on_disk["arrivals"][0]["prefill_length"] == 15659,
               "A: arrivals[] carries the envelope facts");
        // Phase 4 (schema v1): arrivals carry the ingress serial and the
        // frozen queue index; delivery_epoch == delivery_sequence;
        // completed_nodes/retry_items/affected_ranks empty; snapshot_handle
        // self-consistent {epoch, tick, kind:""} (compat-built delta gets
        // the v1 defaults).
        expect(request1_on_disk["arrivals"][0]["ingress_seq"] == 0 &&
                   request1_on_disk["arrivals"][0]["queue_index"] == -1,
               "A: arrivals[] carries ingress_seq/queue_index (v1)");
        expect(request1_on_disk["delivery_epoch"] ==
                   request1_on_disk["delivery_sequence"],
               "A: delivery_epoch == delivery_sequence (v1)");
        expect(request1_on_disk["completed_nodes"].is_array() &&
                   request1_on_disk["completed_nodes"].empty() &&
                   request1_on_disk["retry_items"].is_array() &&
                   request1_on_disk["retry_items"].empty() &&
                   request1_on_disk["affected_ranks"].is_array() &&
                   request1_on_disk["affected_ranks"].empty(),
               "A: completed_nodes/retry_items/affected_ranks empty (v1)");
        expect(request1_on_disk["snapshot_handle"]["epoch"] == 1 &&
                   request1_on_disk["snapshot_handle"]["tick"] == 94835000 &&
                   request1_on_disk["snapshot_handle"]["kind"] == "",
               "A: snapshot_handle self-consistent placeholder (v1)");
        expect(request1_on_disk["completed_groups"][0]["stage"] == "prefill" &&
                   request1_on_disk["completed_groups"][0]["generation"] == 0 &&
                   request1_on_disk["completed_groups"][0]["node_count"] == 5,
               "A: completed_groups[] carries the watch-fire facts");

        expect(batch1.source_delivery_sequence == 1 && batch1.batch_id == 1,
               "A: batch echoes seq 1");
        expect(batch1.error.empty(), "A: no error on the happy path");
        expect(batch1.nodes == nlohmann::json::parse(R"([{"echo_node":1},{"echo_node":2}])"),
               "A: nodes[] arrived intact");
        expect(batch1.parent_edges == nlohmann::json::parse(R"([{"from":0,"to":1}])"),
               "A: parent_edges[] arrived intact");
        expect(batch1.watches.is_array() && batch1.watches.empty(),
               "A: watches[] arrived intact");
        expect(batch1.assignments == nlohmann::json::parse(R"([{"echo":"assignment"}])"),
               "A: assignments[] arrived intact");
        expect(batch1.kv_actions == nlohmann::json::parse(R"([{"echo":"kv_action"}])"),
               "A: kv_actions[] arrived intact");

        // Phase-7 §10.3 中间产物生命周期: read_response CONSUMES AND DELETES
        // the response file right after parsing (fail-closed path retains
        // it). The field-level round-trip fidelity that response_N.json used
        // to carry is now verified upstream -- the echoed nodes/parent_edges/
        // assignments/kv_actions arrived intact in batch1 above, and the
        // Python decision content is covered by the decision_log byte-gate
        // of the full runs. request_N.json is retained (fixture input +
        // audit evidence) and asserted below.
        expect(!file_exists(bridge_dir + "/response_1.json"),
               "A: response_1.json consumed and deleted (phase-7 §10.3)");
        expect(file_exists(bridge_dir + "/request_1.json"),
               "A: request_1.json retained (replay input + audit evidence)");

        // --- delta 2: DECODE_COMPLETION + REQUEST_COMPLETE ---
        DecisionEvent decode_ev;
        decode_ev.reason = DecisionReason::DECODE_COMPLETION;
        decode_ev.request_id = "s0_r0";
        decode_ev.stage = "decode";
        decode_ev.generation = 1;
        decode_ev.payload.watch_member_count = 42;

        DecisionEvent complete_ev;
        complete_ev.reason = DecisionReason::REQUEST_COMPLETE;
        complete_ev.request_id = "s0_r0";
        complete_ev.generation = 1;

        const StateDelta delta2 =
            build_state_delta({decode_ev, complete_ev}, 94835010, 2);

        const GraphBatch batch2 = bridge.deliver_and_receive(delta2);
        expect(batch2.source_delivery_sequence == 2 && batch2.batch_id == 2,
               "B: batch echoes seq 2");
        // Phase-7 §10.3: response consumed and deleted (see part A).
        expect(!file_exists(bridge_dir + "/response_2.json"),
               "B: response_2.json consumed and deleted (phase-7 §10.3)");

        // --- commit acks: seq 1, seq 2, then a DUPLICATE seq 1 ---
        bridge.send_commit_ack(/*batch_id=*/1, /*delivery_seq=*/1, true);
        bridge.send_commit_ack(/*batch_id=*/2, /*delivery_seq=*/2, true);
        bridge.send_commit_ack(/*batch_id=*/1, /*delivery_seq=*/1, true);
    }  // destructor closes req_notify -> Python EOF -> exit 0

    int status = 0;
    ::waitpid(py, &status, 0);
    expect(WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "A: echo Python exited 0 on EOF (clean run end)");

    // ack receipt: exactly two lines (duplicate seq 1 was idempotent)
    std::ifstream receipt(ack_receipt);
    std::string line1, line2, line3;
    std::getline(receipt, line1);
    std::getline(receipt, line2);
    std::getline(receipt, line3);
    expect(!line1.empty() && !line2.empty() && line3.empty(),
           "A: ack receipt has exactly 2 lines (duplicate ignored)");
    expect(line1.find("\"delivery_sequence\": 1") != std::string::npos &&
               line1.find("\"batch_id\": 1") != std::string::npos &&
               line1.find("\"success\": true") != std::string::npos,
           "A: ack line 1 carries seq/batch_id/success");
    expect(line2.find("\"delivery_sequence\": 2") != std::string::npos &&
               line2.find("\"batch_id\": 2") != std::string::npos,
           "A: ack line 2 carries seq/batch_id/success");

    rm_rf(root);
    std::printf("[bridge_loopback_fixture] A: round-trip + ack + clean "
                "exit PASS\n");
}

// ---------------------------------------------------------------- Part B --
// Python crash (EOF on resp_notify) -> C++ aborts with a clear message.
const char* kCrashScript =
    "import os,sys\n"
    "d=sys.argv[1]\n"
    "fd=os.open(d+'/req_notify.fifo', os.O_RDONLY)\n"
    "os.read(fd,1)\n"
    "os.open(d+'/resp_notify.fifo', os.O_WRONLY)\n"
    "sys.exit(0)\n";  // dies WITHOUT writing the response byte

void crash_scenario(const std::string& root) {
    const std::string bridge_dir = root + "/bridge";
    FileDecisionBridge::ensure_bridge_dir(bridge_dir);
    spawn_python3({"-c", kCrashScript, bridge_dir});

    FileDecisionBridge bridge(bridge_dir, /*timeout_ms=*/5000);
    DecisionEvent arrival_ev;
    arrival_ev.reason = DecisionReason::ARRIVAL;
    arrival_ev.request_id = "s0_r0";
    arrival_ev.payload.arrival_world_ns = 1;
    const StateDelta delta =
        build_state_delta({arrival_ev}, 12345, 1);
    (void)bridge.deliver_and_receive(delta);  // must abort inside
    std::fprintf(stderr, "[bridge_loopback_fixture] BUG: crash scenario "
                         "did not abort\n");
    _exit(2);
}

// ---------------------------------------------------------------- Part C --
// Python never responds -> poll timeout -> C++ aborts.
const char* kSleepScript =
    "import os,sys,time\n"
    "d=sys.argv[1]\n"
    "fd=os.open(d+'/req_notify.fifo', os.O_RDONLY)\n"
    "os.read(fd,1)\n"
    "time.sleep(3600)\n";  // never opens resp_notify

void timeout_scenario(const std::string& root) {
    const std::string bridge_dir = root + "/bridge";
    FileDecisionBridge::ensure_bridge_dir(bridge_dir);
    const pid_t py = spawn_python3({"-c", kSleepScript, bridge_dir});

    // let the parent kill the sleeper after the child aborts
    {
        std::ofstream pidfile(root + "/sleep.pid");
        pidfile << py << "\n";
    }

    FileDecisionBridge bridge(bridge_dir, /*timeout_ms=*/1000);
    DecisionEvent arrival_ev;
    arrival_ev.reason = DecisionReason::ARRIVAL;
    arrival_ev.request_id = "s0_r0";
    arrival_ev.payload.arrival_world_ns = 1;
    const StateDelta delta =
        build_state_delta({arrival_ev}, 12345, 1);
    (void)bridge.deliver_and_receive(delta);  // must abort inside
    std::fprintf(stderr, "[bridge_loopback_fixture] BUG: timeout scenario "
                         "did not abort\n");
    _exit(2);
}

// Parent-side driver for the abort scenarios: fork, run in the child (which
// aborts), assert SIGABRT, then kill any leftover sleeper and clean up.
void run_abort_scenario(const char* what, void (*scenario)(const std::string&)) {
    const std::string root = make_temp_root();
    const pid_t child = ::fork();
    if (child == 0) {
        scenario(root);
        _exit(0);  // unreachable: scenario aborts
    }
    int status = 0;
    ::waitpid(child, &status, 0);
    const bool aborted = WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
    expect(aborted, (std::string(what) + " child aborted with SIGABRT").c_str());

    // kill a leftover sleeper from the timeout scenario
    std::ifstream pidfile(root + "/sleep.pid");
    long sleeper = 0;
    if (pidfile >> sleeper && sleeper > 0) {
        ::kill(static_cast<pid_t>(sleeper), SIGKILL);
        ::waitpid(static_cast<pid_t>(sleeper), nullptr, 0);
    }
    rm_rf(root);
    std::printf("[bridge_loopback_fixture] %s PASS\n", what);
}

// Defect-B fix regression (2026-08-16): same driver, but the child's stderr
// is captured to <root>/child.err and the abort MESSAGE is asserted too (the
// fail-closed diagnostics are part of the contract: "which side died" must
// be unambiguous under the long-lived protocol).
void run_abort_scenario_with_message(const char* what, const char* needle,
                                     const char* kill_pid_file,
                                     void (*scenario)(const std::string&)) {
    const std::string root = make_temp_root();
    const pid_t child = ::fork();
    if (child == 0) {
        // capture this child's stderr (bridge_fatal prints there)
        const int errfd =
            ::open((root + "/child.err").c_str(), O_WRONLY | O_CREAT | O_TRUNC,
                   0644);
        if (errfd >= 0) {
            ::dup2(errfd, 2);
        }
        scenario(root);
        _exit(0);  // unreachable: scenario aborts
    }
    int status = 0;
    ::waitpid(child, &status, 0);
    const bool aborted = WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
    expect(aborted, (std::string(what) + " child aborted with SIGABRT").c_str());
    std::ifstream err(root + "/child.err");
    std::stringstream buf;
    buf << err.rdbuf();
    const std::string err_text = buf.str();
    expect(err_text.find(needle) != std::string::npos,
           (std::string(what) + " abort message contains \"").append(needle)
               .append("\" (got: ")
               .append(err_text)
               .append(")")
               .c_str());
    if (kill_pid_file != nullptr) {
        std::ifstream pidfile(root + "/" + kill_pid_file);
        long leftover = 0;
        if (pidfile >> leftover && leftover > 0) {
            ::kill(static_cast<pid_t>(leftover), SIGKILL);
            ::waitpid(static_cast<pid_t>(leftover), nullptr, 0);
        }
    }
    rm_rf(root);
    std::printf("[bridge_loopback_fixture] %s PASS\n", what);
}

// ---------------------------------------------------------------- Part D --
// Defect-B fix (2026-08-16): genuine peer death under the LONG-LIVED resp
// protocol. Python completes exchange 1 normally, then SIGKILLs itself
// before responding to exchange 2. The C++ lifetime read end must see the
// write end close (read==0) and abort with the unambiguous
// "long-lived resp_notify write end closed" diagnostic -- the same
// signature that, under the OLD per-exchange protocol, also fired for a
// live-but-between-opens peer (the F1 false kill).
const char* kKillScript =
    "import os,sys,json\n"
    "d=sys.argv[1]\n"
    "fd=os.open(d+'/req_notify.fifo', os.O_RDONLY)\n"
    "wfd=os.open(d+'/resp_notify.fifo', os.O_WRONLY)\n"
    "os.read(fd,1)\n"
    "resp={'schema_version':1,'batch_id':1,'source_delivery_sequence':1,"
    "'nodes':[],'parent_edges':[],'watches':[],'assignments':[],"
    "'kv_actions':[],'future_alarms':[]}\n"
    "open(d+'/response_1.json.tmp','w').write(json.dumps(resp))\n"
    "os.replace(d+'/response_1.json.tmp', d+'/response_1.json')\n"
    "os.write(wfd, b'\\n')\n"
    "os.read(fd,1)\n"          // commit-ack doorbell (exchange 1)
    "os.read(fd,1)\n"          // request 2 doorbell
    "os.kill(os.getpid(), 9)\n";  // hard death: NO response for seq 2

void kill_scenario(const std::string& root) {
    const std::string bridge_dir = root + "/bridge";
    FileDecisionBridge::ensure_bridge_dir(bridge_dir);
    const pid_t py = spawn_python3({"-c", kKillScript, bridge_dir});
    {
        std::ofstream pidfile(root + "/kill.pid");
        pidfile << py << "\n";
    }
    FileDecisionBridge bridge(bridge_dir, /*timeout_ms=*/10000);
    DecisionEvent arrival_ev;
    arrival_ev.reason = DecisionReason::ARRIVAL;
    arrival_ev.request_id = "s0_r0";
    arrival_ev.payload.arrival_world_ns = 1;
    const StateDelta delta1 = build_state_delta({arrival_ev}, 12345, 1);
    (void)bridge.deliver_and_receive(delta1);  // exchange 1 completes
    bridge.send_commit_ack(1, 1, true);
    arrival_ev.request_id = "s0_r1";
    const StateDelta delta2 = build_state_delta({arrival_ev}, 12346, 2);
    (void)bridge.deliver_and_receive(delta2);  // must abort inside (peer dead)
    std::fprintf(stderr, "[bridge_loopback_fixture] BUG: kill scenario "
                         "did not abort\n");
    _exit(2);
}

// ---------------------------------------------------------------- Part E --
// Defect-B fix guard: the resp channel is strictly 1 byte per delivery
// (backpressure). A Python that writes TWO bytes for one response must trip
// the in-flight invariant ("protocol violation: more than one response
// byte in flight") instead of silently shifting the exchanges.
const char* kTwoByteScript =
    "import os,sys,json,time\n"
    "d=sys.argv[1]\n"
    "fd=os.open(d+'/req_notify.fifo', os.O_RDONLY)\n"
    "wfd=os.open(d+'/resp_notify.fifo', os.O_WRONLY)\n"
    "os.read(fd,1)\n"
    "resp={'schema_version':1,'batch_id':1,'source_delivery_sequence':1,"
    "'nodes':[],'parent_edges':[],'watches':[],'assignments':[],"
    "'kv_actions':[],'future_alarms':[]}\n"
    "open(d+'/response_1.json.tmp','w').write(json.dumps(resp))\n"
    "os.replace(d+'/response_1.json.tmp', d+'/response_1.json')\n"
    "os.write(wfd, b'\\n\\n')\n"  // protocol violation: 2 bytes, 1 delivery
    "time.sleep(3600)\n";         // stay alive: the invariant must fire

void two_byte_scenario(const std::string& root) {
    const std::string bridge_dir = root + "/bridge";
    FileDecisionBridge::ensure_bridge_dir(bridge_dir);
    const pid_t py = spawn_python3({"-c", kTwoByteScript, bridge_dir});
    {
        std::ofstream pidfile(root + "/twobyte.pid");
        pidfile << py << "\n";
    }
    FileDecisionBridge bridge(bridge_dir, /*timeout_ms=*/10000);
    DecisionEvent arrival_ev;
    arrival_ev.reason = DecisionReason::ARRIVAL;
    arrival_ev.request_id = "s0_r0";
    arrival_ev.payload.arrival_world_ns = 1;
    const StateDelta delta = build_state_delta({arrival_ev}, 12345, 1);
    (void)bridge.deliver_and_receive(delta);  // must abort inside
    std::fprintf(stderr, "[bridge_loopback_fixture] BUG: two-byte scenario "
                         "did not abort\n");
    _exit(2);
}

}  // namespace

int main(int argc, char* argv[]) {
    const std::string echo_script =
        argc > 1 ? argv[1]
                 : "sh_test_mesh/workload/llama2_7b_inference/online/verify/"
                   "bridge_echo.py";

    round_trip(echo_script);
    run_abort_scenario("B: crash detection", crash_scenario);
    run_abort_scenario("C: timeout", timeout_scenario);
    // Defect-B fix regressions (2026-08-16): genuine peer death under the
    // long-lived protocol (message asserted) and the 1:1 in-flight guard.
    run_abort_scenario_with_message(
        "D: peer death (SIGKILL) detection",
        "long-lived resp_notify write end closed", "kill.pid", kill_scenario);
    run_abort_scenario_with_message(
        "E: 1:1 response-byte invariant",
        "more than one response byte in flight", "twobyte.pid",
        two_byte_scenario);

    if (!g_ok) {
        std::fprintf(stderr,
                     "[bridge_loopback_fixture] FAIL: see messages above\n");
        return 1;
    }
    std::printf("[bridge_loopback_fixture] ALL PASS: round-trip lossless, "
                "ack idempotent, crash+timeout fail-closed\n");
    return 0;
}
