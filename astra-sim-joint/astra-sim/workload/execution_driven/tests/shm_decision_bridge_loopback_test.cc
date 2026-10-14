/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

shm_decision_bridge_loopback_test.cc -- joint v11 §8 保底：ShmDecisionBridge
的进程内 loopback 单测（不依赖 python3；跨语言面由 bridge_loopback_fixture
覆盖）。

覆盖：
  1. ShmFrameChannel 字节往返：二进制负载（含 NUL 与 MB 级帧）经
     create/publish/attach/try_read 逐字节相等；重复 seq 幂等忽略；空槽
     返回 false；未来 seq 原样上交（tracker 语义）。
  2. 传输层 fail-closed 矩阵：超槽容量抛错、crc 破坏抛错、陈旧世代段
     拒绝、段头魔数损坏拒绝。
  3. 桥级三向字节往返（fork 的 C++ 对端模拟 Python 侧）：request（含
     二进制账本拼接字节——独立重算拼接作 oracle）、response、commit_ack
     全部逐字节相等；重复 ack 幂等；析构清理 /dev/shm 段；stats 口径
     （channel_bytes/forced_flush_count/roundtrip）与旧文件传输计数一致。
  4. 桥级 fail-closed：response 帧 crc 破坏 → "[Error] ... response frame
     corrupt ... checksum" + SIGABRT；request 超槽容量（256MiB+1 账本）
     → "exceeds the shm slot capacity" + SIGABRT。

Build: CMake 目标 AstraSim_Analytical_Congestion_Aware_ShmDecisionBridge
LoopbackTest（execution_driven glob 自动纳入 DecisionBridge.cc /
ShmFrameChannel.cc）。已注册 CTest（ShmDecisionBridgeLoopbackTest）。
*******************************************************************************/

#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <json/json.hpp>

#include "astra-sim/workload/execution_driven/DecisionBridge.hh"
#include "astra-sim/workload/execution_driven/ShmFrameChannel.hh"

using namespace AstraSim::ExecutionDriven;

namespace {

bool g_ok = true;

void expect(bool cond, const std::string& what) {
    if (!cond) {
        std::fprintf(stderr, "[shm_bridge_test] FAIL: %s\n", what.c_str());
        g_ok = false;
    }
}

std::string make_temp_root(const char* tag) {
    const std::string root = "/tmp/sh30_shm_bridge_" + std::string(tag) +
                             "_" + std::to_string(::getpid());
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
        std::fprintf(stderr, "[shm_bridge_test] rm_rf(%s) rc=%d\n",
                     path.c_str(), rc);
    }
}

// ------------------------------------------------------------ test 1 + 2 --
// Channel-layer byte round-trip and the fail-closed matrix.
void channel_tests() {
    const std::string name = "/astra_brg_test_chan_" +
                             std::to_string(::getpid()) + "_" +
                             std::to_string(::time(nullptr));
    const uint64_t gen = new_bridge_shm_generation();

    // --- byte round-trip with a binary payload (NULs + MB scale) ---
    std::string payload;
    payload.reserve(3u << 20);
    for (uint32_t i = 0; i < (3u << 20); ++i) {
        payload.push_back(static_cast<char>((i * 7 + (i >> 9)) & 0xFF));
    }
    ShmFrameChannel writer = ShmFrameChannel::create(name, gen, 8u << 20);
    writer.publish(/*seq=*/7, payload);
    ShmFrameChannel reader = ShmFrameChannel::attach(name, gen, 8u << 20);
    std::string got;
    uint64_t seq_out = 0;
    expect(reader.try_read(7, &got, &seq_out) && seq_out == 7 &&
               got == payload,
           "channel: binary payload round-trips byte-identical");
    expect(!reader.try_read(8, &got, &seq_out),
           "channel: duplicate seq (< expect) is ignored");
    writer.publish(8, std::string("next"));
    expect(reader.try_read(8, &got, &seq_out) && got == "next",
           "channel: next seq after duplicate read");
    writer.close();
    reader.close();
    ::shm_unlink(name.c_str());

    // --- future seq surfaces to the caller (tracker semantics) ---
    const std::string fname = name + "_f";
    ShmFrameChannel w2 = ShmFrameChannel::create(fname, gen, 4096);
    w2.publish(9, std::string("future"));
    ShmFrameChannel r2 = ShmFrameChannel::attach(fname, gen, 4096);
    expect(r2.try_read(5, &got, &seq_out) && seq_out == 9 && got == "future",
           "channel: future seq returns to the caller (tracker fail-closes)");
    w2.close();
    r2.close();
    ::shm_unlink(fname.c_str());

    // --- empty slot ---
    const std::string ename = name + "_e";
    ShmFrameChannel w3 = ShmFrameChannel::create(ename, gen, 4096);
    ShmFrameChannel r3 = ShmFrameChannel::attach(ename, gen, 4096);
    expect(!r3.try_read(0, &got, &seq_out), "channel: empty slot reads false");
    w3.close();
    r3.close();
    ::shm_unlink(ename.c_str());

    // --- oversize fail-closed ---
    const std::string oname = name + "_o";
    ShmFrameChannel w4 = ShmFrameChannel::create(oname, gen, 64);
    bool threw = false;
    try {
        w4.publish(0, std::string(65, 'x'));
    } catch (const std::exception& exc) {
        threw = std::string(exc.what()).find("exceeds") != std::string::npos;
    }
    expect(threw, "channel: oversize frame throws 'exceeds' (fail-closed)");
    w4.close();
    ::shm_unlink(oname.c_str());

    // --- crc corruption fail-closed ---
    const std::string cname = name + "_c";
    ShmFrameChannel w5 = ShmFrameChannel::create(cname, gen, 4096);
    w5.publish(3, std::string("checksum me"));
    {
        // raw poke: flip one payload byte behind the channel's back
        const std::string path = std::string("/dev/shm") + cname;
        int fd = ::open(path.c_str(), O_RDWR);
        void* base = ::mmap(nullptr, 64 + 4096, PROT_READ | PROT_WRITE,
                            MAP_SHARED, fd, 0);
        static_cast<uint8_t*>(base)[64 + 16 + 2] ^= 0xFF;
        ::munmap(base, 64 + 4096);
        ::close(fd);
    }
    ShmFrameChannel r5 = ShmFrameChannel::attach(cname, gen, 4096);
    threw = false;
    try {
        r5.try_read(3, &got, &seq_out);
    } catch (const std::exception& exc) {
        threw = std::string(exc.what()).find("checksum") != std::string::npos;
    }
    expect(threw, "channel: crc corruption throws 'checksum' (fail-closed)");
    w5.close();
    r5.close();
    ::shm_unlink(cname.c_str());

    // --- stale generation rejected ---
    const std::string gname = name + "_g";
    ShmFrameChannel w6 = ShmFrameChannel::create(gname, gen, 4096);
    threw = false;
    try {
        ShmFrameChannel bad = ShmFrameChannel::attach(gname, gen + 1, 4096);
        (void)bad;
    } catch (const std::exception& exc) {
        threw = std::string(exc.what()).find("generation") != std::string::npos;
    }
    expect(threw, "channel: stale generation attach throws (cross-run guard)");
    // --- corrupt segment header rejected ---
    {
        const std::string path = std::string("/dev/shm") + gname;
        int fd = ::open(path.c_str(), O_RDWR);
        void* base = ::mmap(nullptr, 64 + 4096, PROT_READ | PROT_WRITE,
                            MAP_SHARED, fd, 0);
        static_cast<uint8_t*>(base)[0] ^= 0xFF;  // break the segment magic
        ::munmap(base, 64 + 4096);
        ::close(fd);
    }
    threw = false;
    try {
        ShmFrameChannel bad = ShmFrameChannel::attach(gname, gen, 4096);
        (void)bad;
    } catch (const std::exception& exc) {
        threw = std::string(exc.what()).find("not a bridge shm segment") !=
                std::string::npos;
    }
    expect(threw, "channel: corrupt segment header attach throws");
    w6.close();
    ::shm_unlink(gname.c_str());
}

// ------------------------------------------------------------ test 3 + 4 --

// The frozen schema-3 minimal legal response (same shape the cross-language
// fixture scripts emit; kernel_contract mirrors the production default).
nlohmann::json make_minimal_response(const uint64_t seq) {
    nlohmann::json resp;
    resp["schema_version"] = 3;
    resp["batch_id"] = seq;
    resp["source_delivery_sequence"] = seq;
    nlohmann::json contract;
    contract["projection_abi_version"] = 1;
    contract["descriptor_version"] = 1;
    contract["pipeline_protocol_version"] = 1;
    contract["model_version"] = "v7-three-actions-20261014";
    contract["timer_whitelist_version"] = "timer-whitelist-v1-20261005";
    contract["window_tile_limit"] = 1;
    contract["window_local_error_ppm"] = 10000;
    contract["window_refine_margin_ppm"] = 20000;
    contract["prediction_config_sha256"] =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    resp["kernel_contract"] = std::move(contract);
    resp["nodes"] = nlohmann::json::array();
    resp["parent_edges"] = nlohmann::json::array();
    resp["watches"] = nlohmann::json::array();
    resp["assignments"] = nlohmann::json::array();
    resp["kv_actions"] = nlohmann::json::array();
    resp["future_alarms"] = nlohmann::json::array();
    return resp;
}

// Independent oracle for the C++ splice: rebuild the exact request frame
// bytes from (request JSON with null placeholder, ledger bytes) the way the
// OLD write_request_with_ledger did (documented marker splice).
std::string expected_request_bytes(const nlohmann::json& request_with_null,
                                   const std::string& ledger_bytes) {
    const std::string dumped = request_with_null.dump();
    const std::string marker = "\"work_ledger\":null";
    const std::size_t at = dumped.find(marker);
    if (at == std::string::npos) {
        return std::string();
    }
    std::string out;
    out.reserve(dumped.size() + ledger_bytes.size());
    out.append(dumped, 0, at + marker.size() - 4);
    out.append(ledger_bytes);
    out.append(dumped, at + marker.size(), std::string::npos);
    return out;
}

struct PeerFacts {
    std::string bridge_dir;
    std::string req_segment;
    std::string ack_segment;
    std::string resp_segment;
    uint64_t generation = 0;
    uint64_t capacity = 0;
};

// The forked "Python side": one loop mirroring BridgeServer.serve_forever --
// blocking fifo byte, ack frame first, then request frame; verifies every
// received byte against the independently computed oracle; publishes the
// response frame + doorbell byte.
[[noreturn]] void peer_main(const PeerFacts& facts,
                            const std::vector<std::string>& want_requests,
                            const std::vector<std::string>& want_acks) {
    bool ok = true;
    ShmFrameChannel req = ShmFrameChannel::attach(
        "/" + facts.req_segment, facts.generation, facts.capacity);
    ShmAckRing ack = ShmAckRing::attach(
        "/" + facts.ack_segment, facts.generation, kBridgeShmAckSlotCount,
        kBridgeShmAckSlotBytes);
    ShmFrameChannel resp = ShmFrameChannel::attach(
        "/" + facts.resp_segment, facts.generation, facts.capacity);
    const int rfd = ::open((facts.bridge_dir + "/req_notify.fifo").c_str(),
                           O_RDONLY);  // blocks until the C++ write end
    if (rfd < 0) {
        _exit(3);
    }
    const int wfd = ::open((facts.bridge_dir + "/resp_notify.fifo").c_str(),
                           O_WRONLY);
    if (wfd < 0) {
        _exit(3);
    }
    uint64_t next_req = 0;
    uint64_t next_ack = 0;
    uint64_t ack_tail = 0;
    while (true) {
        char byte = 0;
        const ssize_t n = ::read(rfd, &byte, 1);
        if (n == 0) {
            break;  // C++ destructor closed its write end: clean EOF
        }
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            ok = false;
            break;
        }
        // ack frames first (serve-loop order): drain the ring, skipping
        // duplicates (seq < tracker) -- the coalesced multi-ack delivery
        // the old loose-file probe loop performed.
        for (const auto& entry : ack.drain(&ack_tail)) {
            const uint64_t aseq = entry.first;
            const std::string& ack_payload = entry.second;
            if (aseq < next_ack) {
                continue;  // idempotent duplicate (e.g. the resent ack 0)
            }
            if (aseq == next_ack && aseq < want_acks.size() &&
                ack_payload == want_acks[aseq]) {
                next_ack = aseq + 1;
            } else {
                std::fprintf(stderr, "[peer] ack mismatch aseq=%llu want=%u "
                                     "bytes=%zu/%zu\n",
                             (unsigned long long)aseq,
                             (unsigned)want_acks.size(),
                             ack_payload.size(),
                             aseq < want_acks.size()
                                 ? want_acks[aseq].size() : 0);
                ok = false;
            }
        }
        // request frames
        std::string req_payload;
        uint64_t rseq = 0;
        if (req.try_read(next_req, &req_payload, &rseq)) {
            if (rseq == next_req && rseq < want_requests.size() &&
                req_payload == want_requests[rseq]) {
                resp.publish(rseq, make_minimal_response(rseq).dump());
                const ssize_t w = ::write(wfd, "\n", 1);
                if (w != 1) {
                    ok = false;
                }
                next_req = rseq + 1;
            } else {
                std::fprintf(stderr, "[peer] req mismatch rseq=%llu "
                                     "next=%llu size=%zu want=%zu\n",
                             (unsigned long long)rseq,
                             (unsigned long long)next_req,
                             req_payload.size(),
                             rseq < want_requests.size()
                                 ? want_requests[rseq].size() : 0);
                ok = false;
            }
        }
        // neither frame present = duplicate doorbell (duplicate ack) --
        // idempotent ignore, exactly like BridgeServer's tracker filter.
    }
    if (next_req != want_requests.size()) {
        std::fprintf(stderr, "[peer] EOF at next_req=%llu of %zu\n",
                     (unsigned long long)next_req, want_requests.size());
        ok = false;
    }
    // duplicates must have been ignored, not re-processed
    if (next_ack != 2) {
        std::fprintf(stderr, "[peer] next_ack=%llu at EOF\n",
                     (unsigned long long)next_ack);
        ok = false;
    }
    req.close();
    ack.close();
    resp.close();
    ::close(rfd);
    ::close(wfd);
    _exit(ok ? 0 : 5);
}

void bridge_roundtrip(const char* tag, const bool with_ledger) {
    const std::string root = make_temp_root(tag);
    const std::string bridge_dir = root + "/bridge";
    ShmDecisionBridge::ensure_bridge_dir(bridge_dir);

    // Deltas + ledger bytes fixed BEFORE the fork so both sides see the
    // same values; the oracle (child) recomputes the spliced request bytes
    // independently of the bridge's own assembly.
    DecisionEvent arrival_ev;
    arrival_ev.reason = DecisionReason::ARRIVAL;
    arrival_ev.request_id = "s0_r0";
    arrival_ev.payload.session_id = "s0";
    arrival_ev.payload.turn_index = 0;
    arrival_ev.payload.prefill_length = 15659;
    arrival_ev.payload.decode_length = 202;
    arrival_ev.payload.arrival_world_ns = 94835000;
    const StateDelta delta1 = build_state_delta({arrival_ev}, 94835000, 0);
    DecisionEvent decode_ev;
    decode_ev.reason = DecisionReason::DECODE_COMPLETION;
    decode_ev.request_id = "s0_r0";
    decode_ev.stage = "decode";
    decode_ev.generation = 1;
    decode_ev.payload.watch_member_count = 42;
    const StateDelta delta2 =
        build_state_delta({decode_ev}, 94835010, 1);

    std::string ledger_bytes;
    if (with_ledger) {
        // binary blob: NULs, high bytes, >64KiB (compact-frame scale)
        ledger_bytes.assign(100000, '\0');
        for (size_t i = 0; i < ledger_bytes.size(); ++i) {
            ledger_bytes[i] = static_cast<char>((i * 31 + (i >> 7)) & 0xFF);
        }
    }

    // Expected request bytes (oracle input) and ack dumps.
    std::vector<std::string> want_requests;
    for (const StateDelta* delta : {&delta1, &delta2}) {
        nlohmann::json request = build_request_json(*delta);
        if (with_ledger) {
            request["work_ledger"] = nullptr;
            want_requests.push_back(
                expected_request_bytes(request, ledger_bytes));
        } else {
            want_requests.push_back(request.dump());
        }
    }
    std::vector<std::string> want_acks;
    {
        nlohmann::json ack0;
        ack0["schema_version"] = kDecisionBridgeSchemaVersion;
        ack0["delivery_sequence"] = 0;
        ack0["batch_id"] = 0;
        ack0["success"] = true;
        want_acks.push_back(ack0.dump());
        nlohmann::json ack1;
        ack1["schema_version"] = kDecisionBridgeSchemaVersion;
        ack1["delivery_sequence"] = 1;
        ack1["batch_id"] = 1;
        ack1["success"] = true;
        want_acks.push_back(ack1.dump());
    }
    const std::vector<std::string> want_responses = {
        make_minimal_response(0).dump(), make_minimal_response(1).dump()};

    pid_t peer = -1;
    bool batches_ok = false;
    uint64_t got_channel_bytes = 0;
    uint64_t got_flush_count = 0;
    uint64_t got_roundtrip_count = 0;
    {
        ShmDecisionBridge bridge(bridge_dir, /*timeout_ms=*/10000,
                                 /*num_ranks=*/1);
        if (with_ledger) {
            bridge.set_work_ledger_provider(
                [&ledger_bytes](const StateDelta&) {
                    return WorkLedgerWirePayload{ledger_bytes};
                });
        }
        PeerFacts facts;
        facts.bridge_dir = bridge_dir;
        facts.req_segment = bridge.shm_request_segment();
        facts.ack_segment = bridge.shm_commit_ack_segment();
        facts.resp_segment = bridge.shm_response_segment();
        facts.generation = bridge.shm_generation();
        facts.capacity = bridge.shm_slot_capacity();

        peer = ::fork();
        if (peer == 0) {
            peer_main(facts, want_requests, want_acks);
        }

        bridge.open_notify();
        const GraphBatch batch1 = bridge.deliver_and_receive(delta1);
        const GraphBatch batch2 = bridge.deliver_and_receive(delta2);
        batches_ok = batch1.source_delivery_sequence == 0 &&
                     batch1.error.empty() &&
                     batch2.source_delivery_sequence == 1;
        bridge.send_commit_ack(0, 0, true);
        bridge.send_commit_ack(1, 1, true);
        bridge.send_commit_ack(0, 0, true);  // duplicate: must be idempotent

        // Stats accounting keeps the old file-transport semantics: request +
        // response + (duplicate-including) ack payload bytes; one flush per
        // published frame on the C++ side (2 requests + 3 acks).
        got_channel_bytes = bridge.stats().channel_bytes;
        got_flush_count = bridge.stats().forced_flush_count;
        got_roundtrip_count = bridge.stats().roundtrip_count;
    }  // destructor FIRST: closes the req write end -> peer sees EOF and
       // exits (the peer loop only ends on EOF, mirroring serve_forever)

    expect(batches_ok,
           std::string(tag) + ": batches parsed from the response frames");
    uint64_t want_bytes = 0;
    for (const std::string& b : want_requests) {
        want_bytes += b.size();
    }
    for (const std::string& b : want_responses) {
        want_bytes += b.size();
    }
    want_bytes += 2 * want_acks[0].size() + want_acks[1].size();
    expect(got_channel_bytes == want_bytes,
           std::string(tag) + ": channel_bytes == payload byte total");
    expect(got_flush_count == 5,
           std::string(tag) + ": forced_flush_count == 5 (2 req + 3 ack)");
    expect(got_roundtrip_count == 2,
           std::string(tag) + ": roundtrip_count == 2");

    int status = 0;
    ::waitpid(peer, &status, 0);
    expect(WIFEXITED(status) && WEXITSTATUS(status) == 0,
           std::string(tag) + ": peer verified every frame byte-identical");
}

// Bridge-level fail-closed: response frame with a broken checksum.
void response_crc_scenario(const std::string& root) {
    const std::string bridge_dir = root + "/bridge";
    ShmDecisionBridge::ensure_bridge_dir(bridge_dir);
    ShmDecisionBridge bridge(bridge_dir, /*timeout_ms=*/10000,
                             /*num_ranks=*/1);
    const std::string resp_segment = bridge.shm_response_segment();
    const uint64_t gen = bridge.shm_generation();
    const uint64_t cap = bridge.shm_slot_capacity();
    const pid_t peer = ::fork();
    if (peer == 0) {
        // "Python side" publishes a valid frame, then corrupts one payload
        // byte behind the channel's back, then rings the doorbell.
        ShmFrameChannel resp =
            ShmFrameChannel::attach("/" + resp_segment, gen, cap);
        resp.publish(0, make_minimal_response(0).dump());
        resp.close();
        const std::string path = std::string("/dev/shm/") + resp_segment;
        const int fd = ::open(path.c_str(), O_RDWR);
        void* base = ::mmap(nullptr, 64 + cap, PROT_READ | PROT_WRITE,
                            MAP_SHARED, fd, 0);
        static_cast<uint8_t*>(base)[64 + 16 + 5] ^= 0xFF;
        ::munmap(base, 64 + cap);
        ::close(fd);
        const int rfd =
            ::open((bridge_dir + "/req_notify.fifo").c_str(), O_RDONLY);
        const int wfd =
            ::open((bridge_dir + "/resp_notify.fifo").c_str(), O_WRONLY);
        char bell = 0;
        if (::read(rfd, &bell, 1) == 1) {
            if (::write(wfd, "\n", 1) == 1) {
                ::sleep(3600);  // stay alive: C++ must fail closed
            }
        }
        _exit(0);
    }
    {
        std::ofstream pidfile(root + "/peer.pid");
        pidfile << peer << "\n";
    }
    DecisionEvent arrival_ev;
    arrival_ev.reason = DecisionReason::ARRIVAL;
    arrival_ev.request_id = "s0_r0";
    arrival_ev.payload.arrival_world_ns = 1;
    const StateDelta delta = build_state_delta({arrival_ev}, 12345, 0);
    (void)bridge.deliver_and_receive(delta);  // aborts inside
    _exit(2);
}

// Bridge-level fail-closed: a ledger payload larger than the 256MiB slot.
void request_oversize_scenario(const std::string& root) {
    const std::string bridge_dir = root + "/bridge";
    ShmDecisionBridge::ensure_bridge_dir(bridge_dir);
    ShmDecisionBridge bridge(bridge_dir, /*timeout_ms=*/10000,
                             /*num_ranks=*/1);
    const std::string big(kBridgeShmSlotCapacity + 1, 'L');
    bridge.set_work_ledger_provider([&big](const StateDelta&) {
        return WorkLedgerWirePayload{big};
    });
    DecisionEvent arrival_ev;
    arrival_ev.reason = DecisionReason::ARRIVAL;
    arrival_ev.request_id = "s0_r0";
    arrival_ev.payload.arrival_world_ns = 1;
    const StateDelta delta = build_state_delta({arrival_ev}, 12345, 0);
    (void)bridge.deliver_and_receive(delta);  // aborts inside
    _exit(2);
}

void run_abort_scenario(const char* what, const char* needle,
                        void (*scenario)(const std::string&)) {
    const std::string root = make_temp_root(what);
    const pid_t child = ::fork();
    if (child == 0) {
        const int errfd =
            ::open((root + "/child.err").c_str(), O_WRONLY | O_CREAT | O_TRUNC,
                   0644);
        if (errfd >= 0) {
            ::dup2(errfd, 2);
        }
        scenario(root);
        _exit(0);
    }
    int status = 0;
    ::waitpid(child, &status, 0);
    expect(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT,
           std::string(what) + " aborts with SIGABRT");
    std::ifstream err(root + "/child.err");
    std::stringstream buf;
    buf << err.rdbuf();
    const std::string text = buf.str();
    expect(text.find("[Error] (execution_driven/bridge)") !=
                   std::string::npos &&
               text.find(needle) != std::string::npos,
           std::string(what) + " carries the bridge fatal line");
    // kill a leftover peer
    std::ifstream pidfile(root + "/peer.pid");
    long leftover = 0;
    if (pidfile >> leftover && leftover > 0) {
        ::kill(static_cast<pid_t>(leftover), SIGKILL);
        ::waitpid(static_cast<pid_t>(leftover), nullptr, 0);
    }
    // The aborting child never ran its destructor, so its segments remain
    // in /dev/shm (deliberate post-mortem semantics); reclaim them here via
    // the retained bridge_shm.json pointer.
    std::ifstream info((root + "/bridge/bridge_shm.json").c_str());
    if (info.good()) {
        try {
            const nlohmann::json parsed = nlohmann::json::parse(info);
            for (const char* key :
                 {"request_segment", "commit_ack_segment", "response_segment"}) {
                const std::string name = "/" + parsed.at(key).get<std::string>();
                ::shm_unlink(name.c_str());
            }
        } catch (const std::exception&) {
            // best effort only
        }
    }
    rm_rf(root);
    std::printf("[shm_bridge_test] %s PASS\n", what);
}

void bridge_teardown_cleanup() {
    // After the destructor the three segments must be gone from /dev/shm;
    // bridge_shm.json deliberately remains (post-mortem pointer, atomically
    // rewritten by the next ensure_bridge_dir).
    const std::string root = make_temp_root("teardown");
    const std::string bridge_dir = root + "/bridge";
    ShmDecisionBridge::ensure_bridge_dir(bridge_dir);
    std::string names[3];
    {
        ShmDecisionBridge bridge(bridge_dir);
        names[0] = bridge.shm_request_segment();
        names[1] = bridge.shm_commit_ack_segment();
        names[2] = bridge.shm_response_segment();
    }  // destructor: unmap + shm_unlink x3
    for (const std::string& name : names) {
        const std::string path = std::string("/dev/shm/") + name;
        struct stat st {};
        expect(::stat(path.c_str(), &st) != 0 && errno == ENOENT,
               "teardown: segment unlinked from /dev/shm (" + path + ")");
    }
    struct stat info_st {};
    expect(::stat((bridge_dir + "/bridge_shm.json").c_str(), &info_st) == 0,
           "teardown: bridge_shm.json pointer retained (post-mortem)");
    rm_rf(root);
}

}  // namespace

int main() {
    channel_tests();
    bridge_roundtrip("plain", /*with_ledger=*/false);
    bridge_roundtrip("ledger", /*with_ledger=*/true);
    run_abort_scenario("response crc corruption",
                       "response frame corrupt", response_crc_scenario);
    run_abort_scenario("request oversize",
                       "exceeds the shm slot capacity",
                       request_oversize_scenario);
    bridge_teardown_cleanup();

    if (!g_ok) {
        std::fprintf(stderr, "[shm_bridge_test] FAIL: see messages above\n");
        return 1;
    }
    std::printf("[shm_bridge_test] ALL PASS: channel bytes round-trip, "
                "bridge 3-way byte-identical, fail-closed matrix, teardown\n");
    return 0;
}
