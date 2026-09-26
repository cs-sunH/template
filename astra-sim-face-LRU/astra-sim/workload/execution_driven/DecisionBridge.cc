/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

DecisionBridge -- execution-driven mechanism layer (wscllm phase 1).
File-implementation of the bridge protocol v1 (方案 §4 步骤 1-7).

Fail-closed semantics: every protocol violation (Python crash via EOF,
EPIPE on the notification write, poll timeout, missing/corrupt response,
schema mismatch, non-empty error field) prints one [Error] line and aborts.
The static binary never constructs a bridge (zero impact on the baseline).
*******************************************************************************/

#include "astra-sim/workload/execution_driven/DecisionBridge.hh"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <poll.h>
#include <sstream>
#include <string>
#include <thread>

namespace AstraSim {
namespace ExecutionDriven {

namespace {

[[noreturn]] void bridge_fatal(const std::string& what) {
    std::fprintf(stderr,
                 "[Error] (execution_driven/bridge) %s\n", what.c_str());
    std::abort();
}

std::string reason_name(const DecisionReason reason) {
    switch (reason) {
        case DecisionReason::ARRIVAL:
            return "ARRIVAL";
        case DecisionReason::PREFILL_DRAIN:
            return "PREFILL_DRAIN";
        case DecisionReason::DECODE_COMPLETION:
            return "DECODE_COMPLETION";
        case DecisionReason::REQUEST_COMPLETE:
            return "REQUEST_COMPLETE";
    }
    return "UNKNOWN";
}

// Recursive mkdir(2) loop (no std::filesystem dependency).
void mkdir_p(const std::string& path) {
    std::string cur;
    for (const char* p = path.c_str(); *p != '\0'; ++p) {
        cur += *p;
        if (*p == '/') {
            if (::mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) {
                bridge_fatal("mkdir " + cur + ": " + std::strerror(errno));
            }
        }
    }
    if (::mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) {
        bridge_fatal("mkdir " + cur + ": " + std::strerror(errno));
    }
}

void ensure_fifo(const std::string& path) {
    if (::mkfifo(path.c_str(), 0600) != 0) {
        if (errno != EEXIST) {
            bridge_fatal("mkfifo " + path + ": " + std::strerror(errno));
        }
        struct stat st;
        if (::stat(path.c_str(), &st) != 0 || !S_ISFIFO(st.st_mode)) {
            bridge_fatal(path + " exists but is not a FIFO");
        }
    }
}

bool file_name_matches(const std::string& name, const char* prefix) {
    const size_t plen = std::strlen(prefix);
    if (name.size() <= plen || name.compare(0, plen, prefix) != 0) {
        return false;
    }
    return name.compare(name.size() - 5, 5, ".json") == 0 ||
           name.compare(name.size() - 4, 4, ".tmp") == 0;
}

nlohmann::json read_json_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    nlohmann::json value;
    in >> value;
    if (in.bad()) {
        throw std::runtime_error("corrupt JSON in " + path);
    }
    return value;
}

// poll() helper: 0 = timeout, 1 = readable (POLLIN or POLLHUP), -1 = error.
// timeout_ms == 0 means wait forever.
int wait_readable(int fd, int timeout_ms) {
    struct pollfd pfd;
    pfd.fd = fd;
    pfd.events = POLLIN | POLLHUP;
    pfd.revents = 0;
    const int ms = (timeout_ms == 0) ? -1 : timeout_ms;
    const int r = ::poll(&pfd, 1, ms);
    if (r == 0) {
        return 0;
    }
    if (r < 0) {
        return -1;
    }
    return 1;
}

}  // namespace

// ------------------------------------------------------------- serializer --

nlohmann::json build_request_json(const StateDelta& delta) {
    nlohmann::json req;
    req["schema_version"] = kDecisionBridgeSchemaVersion;
    req["delivery_sequence"] = delta.delivery_sequence;
    // Phase 4 (v1): the delivery-epoch counter (== delivery_sequence in v1;
    // the Python validator asserts equality).
    req["delivery_epoch"] = delta.delivery_epoch;
    req["tick"] = delta.tick;
    // Step 1-11: the explicit T->T+1 deferral record (0 = same-tick delivery).
    req["deferred_from_tick"] = delta.deferred_from_tick;
    req["reasons"] = nlohmann::json::array();
    req["arrivals"] = nlohmann::json::array();
    req["completed_groups"] = nlohmann::json::array();
    // Phase 4: preserve the schema field on every delivery. Production drains
    // this as [] and keeps only O(1) run-lifetime terminal counters; the
    // explicit ASTRA_SIM_ONLINE_COMPLETED_NODES=exact audit mode restores the
    // frozen per-node {rank,node_id,tick} records.
    req["completed_nodes"] = nlohmann::json::array();
    for (const auto& fact : delta.completed_nodes) {
        nlohmann::json f;
        f["request_id"] = fact.request_id;
        f["stage"] = fact.stage;
        f["generation"] = fact.generation;
        f["terminal_status"] = fact.terminal_status;
        f["rank"] = fact.rank;
        f["node_id"] = fact.node_id;
        f["tick"] = fact.tick;
        req["completed_nodes"].push_back(std::move(f));
    }
    // Phase 4 (v1): the affected-rank set (union of the epoch's
    // completed_groups member ranks, sorted unique).
    req["affected_ranks"] = nlohmann::json::array();
    for (const int rank : delta.affected_ranks) {
        req["affected_ranks"].push_back(rank);
    }
    // Phase 4 (v1): the placeholder snapshot handle (expiry rule frozen in
    // the contract -- valid only in the epoch/tick it was created in).
    req["snapshot_handle"] = {
        {"epoch", delta.snapshot_handle.epoch},
        {"tick", delta.snapshot_handle.tick},
        {"kind", delta.snapshot_handle.kind},
    };
    // v0 reserved field, kept for phase-1 backward comprehension (always an
    // empty object in v1; snapshot_handle is the v1 successor).
    req["snapshot"] = nlohmann::json::object();
    // Phase-3 sensing (方案 §6.2 操作 1): the two-layer remaining-load query,
    // C++ side -- per-rank injected-unfinished ledger summary. Always
    // serialized (empty array when --sensing-enabled is off) so the schema is
    // stable across sensing on/off runs; the summary is delivered in the
    // StateDelta and carried here verbatim. Query/audit data only -- the
    // wscllm strategy's red-line decision inputs never consume it.
    nlohmann::json injected = nlohmann::json::array();
    for (const auto& rs : delta.injected_unfinished) {
        nlohmann::json rank_json;
        rank_json["rank"] = rs.rank;
        rank_json["node_count"] = rs.node_count;
        rank_json["compute_ops"] = rs.compute_ops;
        rank_json["comm_bytes"] = rs.comm_bytes;
        rank_json["estimated_remaining_ns"] = rs.estimated_remaining_ns;
        rank_json["in_flight_node_count"] = rs.in_flight_node_count;
        rank_json["in_flight_gpu_ops"] = rs.in_flight_gpu_ops;
        rank_json["free_node_count"] = rs.free_node_count;
        rank_json["per_request"] = nlohmann::json::array();
        for (const auto& entry : rs.per_request) {
            nlohmann::json entry_json;
            entry_json["request_id"] = entry.request_id;
            entry_json["stage"] = entry.stage;
            entry_json["generation"] = entry.generation;
            entry_json["node_count"] = entry.node_count;
            entry_json["compute_ops"] = entry.compute_ops;
            entry_json["comm_bytes"] = entry.comm_bytes;
            entry_json["estimated_remaining_ns"] = entry.estimated_remaining_ns;
            rank_json["per_request"].push_back(std::move(entry_json));
        }
        injected.push_back(std::move(rank_json));
    }
    req["ledger_summary"] = {{"injected_unfinished", std::move(injected)}};
    for (const auto& ev : delta.events) {
        req["reasons"].push_back(reason_name(ev.reason));
        switch (ev.reason) {
            case DecisionReason::ARRIVAL: {
                nlohmann::json a;
                a["request_id"] = ev.request_id;
                a["session_id"] = ev.payload.session_id;
                a["turn_index"] = ev.payload.turn_index;
                a["prefill_length"] = ev.payload.prefill_length;
                a["decode_length"] = ev.payload.decode_length;
                a["inter_request_interval_ns"] =
                    ev.payload.inter_request_interval_ns;
                a["arrival_world_ns"] = ev.payload.arrival_world_ns;
                // Phase 4 (v1): ingress serial + frozen queue index
                // (原契约 §5.1,文档已删除).
                a["ingress_seq"] = ev.payload.ingress_seq;
                a["queue_index"] = ev.payload.queue_index;
                req["arrivals"].push_back(std::move(a));
                break;
            }
            case DecisionReason::PREFILL_DRAIN:
            case DecisionReason::DECODE_COMPLETION:
            case DecisionReason::REQUEST_COMPLETE: {
                nlohmann::json g;
                g["request_id"] = ev.request_id;
                g["stage"] = ev.stage;
                g["generation"] = ev.generation;
                g["node_count"] = ev.payload.watch_member_count;
                req["completed_groups"].push_back(std::move(g));
                break;
            }
        }
    }
    // Phase 4 (schema v1): the frozen queue order -- arrivals[] is sorted by
    // (queue_index, ingress_seq) ascending, deterministically, regardless of
    // the mailbox drain order (alarm order). The Python validator asserts
    // the ascending order (fail-closed; 原契约 §3.3/§6,文档已删除).
    // queue_index is unique within one batch (a request arrives
    // once per epoch), so the sort is total.
    std::sort(req["arrivals"].begin(), req["arrivals"].end(),
              [](const nlohmann::json& a, const nlohmann::json& b) {
                  const int64_t qa = a.value("queue_index", int64_t(-1));
                  const int64_t qb = b.value("queue_index", int64_t(-1));
                  if (qa != qb) {
                      return qa < qb;
                  }
                  return a.value("ingress_seq", uint64_t(0)) <
                         b.value("ingress_seq", uint64_t(0));
              });
    return req;
}

// -------------------------------------------------------------- FileBridge --

FileDecisionBridge::FileDecisionBridge(std::string bridge_dir,
                                       const int timeout_ms,
                                       const int num_ranks)
    : bridge_dir_(std::move(bridge_dir)),
      timeout_ms_(timeout_ms),
      num_ranks_(num_ranks) {}

FileDecisionBridge::~FileDecisionBridge() {
    if (req_notify_fd_ >= 0) {
        // Closing the write end gives Python a clean EOF at run end.
        ::close(req_notify_fd_);
    }
    if (resp_notify_fd_ >= 0) {
        // Defect-B fix: symmetric teardown of the long-lived read end (a
        // Python still blocked writing its next response byte then gets a
        // clean EPIPE instead of an eternal block).
        ::close(resp_notify_fd_);
    }
}

void FileDecisionBridge::ensure_bridge_dir(const std::string& bridge_dir) {
    mkdir_p(bridge_dir);
    ensure_fifo(bridge_dir + "/req_notify.fifo");
    ensure_fifo(bridge_dir + "/resp_notify.fifo");
    // Clean stale files from a previous run in the same directory.
    DIR* dir = ::opendir(bridge_dir.c_str());
    if (dir == nullptr) {
        bridge_fatal("opendir " + bridge_dir + ": " + std::strerror(errno));
    }
    while (struct dirent* entry = ::readdir(dir)) {
        const std::string name = entry->d_name;
        if (file_name_matches(name, "request_") ||
            file_name_matches(name, "response_") ||
            file_name_matches(name, "commit_ack_")) {
            ::unlink((bridge_dir + "/" + name).c_str());
        }
    }
    ::closedir(dir);
}

std::string FileDecisionBridge::request_path(const uint64_t seq) const {
    return bridge_dir_ + "/request_" + std::to_string(seq) + ".json";
}

std::string FileDecisionBridge::response_path(const uint64_t seq) const {
    return bridge_dir_ + "/response_" + std::to_string(seq) + ".json";
}

std::string FileDecisionBridge::ack_path(const uint64_t seq) const {
    return bridge_dir_ + "/commit_ack_" + std::to_string(seq) + ".json";
}

void FileDecisionBridge::write_file_atomic(const std::string& path,
                                           const nlohmann::json& payload) const {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp);
        if (!out) {
            bridge_fatal("cannot open " + tmp);
        }
        // Phase 6 (方案 §9.1): channel-byte and forced-flush accounting. The
        // dump is the exact on-disk payload, so the dump size equals the file
        // bytes; one flush per atomic write (O(deliveries), never O(nodes)).
        const std::string body = payload.dump();
        stats_.channel_bytes += body.size();
        stats_.forced_flush_count += 1;
        out << body;
        out.flush();
        if (!out) {
            bridge_fatal("write failed for " + tmp);
        }
    }
    if (::rename(tmp.c_str(), path.c_str()) != 0) {
        bridge_fatal("rename " + tmp + " -> " + path + ": " +
                     std::strerror(errno));
    }
}

void FileDecisionBridge::open_notify() {
    if (req_notify_fd_ < 0) {
        // O_WRONLY|O_NONBLOCK on a FIFO with no reader fails with ENXIO, so
        // retry until the Python reader's blocking open lands (bounded by
        // timeout_ms_ when > 0). Step 1-10: this runs at startup
        // (main_online.cc) so that even a zero-delivery run holds the write
        // end -- Python's blocking read then sees a clean EOF when the
        // destructor closes it at run end.
        const std::string path = bridge_dir_ + "/req_notify.fifo";
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(timeout_ms_);
        while (true) {
            req_notify_fd_ = ::open(path.c_str(), O_WRONLY | O_NONBLOCK);
            if (req_notify_fd_ >= 0) {
                break;
            }
            if (errno != ENXIO) {
                bridge_fatal("open " + path + ": " + std::strerror(errno));
            }
            if (timeout_ms_ > 0 &&
                std::chrono::steady_clock::now() >= deadline) {
                bridge_fatal("timeout waiting for Python to open " + path);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    // Defect-B fix (2026-08-16): open the resp_notify READ end once and hold
    // it for the bridge lifetime (mirrors the req write end above). A
    // nonblocking read open of a FIFO succeeds immediately, writer or not;
    // Python opens its lifetime write end right after its req read end open
    // (decision_bridge.py serve_forever), so the two sides pair without a
    // handshake. Per-exchange open/close on this channel is what raced: a
    // fresh read fd registered in poll exactly when the writer's 1->0 close
    // transition landed with an empty buffer woke with POLLHUP/read()==0
    // (the false "Python side crashed", F1) and the mirror race EPIPE'd
    // Python's write (F7-style handshake wedges). With both fds long-lived
    // there are no mid-run writer/reader transitions at all: EOF on this fd
    // can only be the peer process dying.
    if (resp_notify_fd_ < 0) {
        const std::string path = bridge_dir_ + "/resp_notify.fifo";
        resp_notify_fd_ = ::open(path.c_str(), O_RDONLY | O_NONBLOCK);
        if (resp_notify_fd_ < 0) {
            bridge_fatal("open " + path + ": " + std::strerror(errno));
        }
    }
}

void FileDecisionBridge::wait_response_byte(const uint64_t seq) {
    // Defect-B fix: one byte off the long-lived read end, with an EAGAIN
    // retry loop (the fd is O_NONBLOCK; a poll wake without a consumable
    // byte -- e.g. POLLHUP edges while Python churns its user-space state
    // -- must not be mistaken for data or death).
    while (true) {
        const int wait = wait_readable(resp_notify_fd_, timeout_ms_);
        if (wait == 0) {
            bridge_fatal("timeout waiting for the Python response byte "
                         "(delivery_sequence=" + std::to_string(seq) + ")");
        }
        if (wait < 0) {
            bridge_fatal("poll resp_notify.fifo: " +
                         std::string(std::strerror(errno)));
        }
        char byte = 0;
        const ssize_t n = ::read(resp_notify_fd_, &byte, 1);
        if (n == 1) {
            // Defect-B fix guard (protocol drift detection): the channel is
            // strictly 1 byte per delivery (backpressure: one in-flight
            // epoch at a time -- C++ never sends request N+1 before
            // response N was consumed, and Python writes exactly one byte
            // per response). A SECOND readable byte means the invariant
            // broke (e.g. a future duplicate notify); fail closed now
            // instead of letting the desync silently shift exchanges.
            char extra = 0;
            const ssize_t m = ::read(resp_notify_fd_, &extra, 1);
            if (m == 1) {
                bridge_fatal("protocol violation: more than one response "
                             "byte in flight (delivery_sequence=" +
                             std::to_string(seq) + "; backpressure is "
                             "1:1)");
            }
            return;  // the notification byte (payload is the response file)
        }
        if (n == 0) {
            // The ONLY way the peer's lifetime write end closes: the Python
            // process died (crash/exit). Genuine crash detection -- under
            // the old per-exchange protocol this same signature also fired
            // for a live-but-between-opens peer (defect B's false kill).
            bridge_fatal("Python side died (its long-lived resp_notify "
                         "write end closed; EOF on resp_notify.fifo, "
                         "delivery_sequence=" +
                         std::to_string(seq) + ")");
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            continue;  // spurious wake: re-enter the poll
        }
        bridge_fatal("read resp_notify.fifo: " +
                     std::string(std::strerror(errno)));
    }
}

void FileDecisionBridge::notify_python() {
    if (req_notify_fd_ < 0) {
        // Defensive fallback: a deliver_and_receive that somehow ran before
        // open_notify() still opens the write end lazily.
        open_notify();
    }
    // Ignore SIGPIPE around the write so a dead Python surfaces as EPIPE
    // (fail-closed message) instead of silently killing the simulator.
    struct sigaction old_action;
    struct sigaction ignore_action;
    std::memset(&ignore_action, 0, sizeof(ignore_action));
    ignore_action.sa_handler = SIG_IGN;
    ::sigemptyset(&ignore_action.sa_mask);
    ::sigaction(SIGPIPE, &ignore_action, &old_action);
    const char byte = '\n';
    const ssize_t n = ::write(req_notify_fd_, &byte, 1);
    ::sigaction(SIGPIPE, &old_action, nullptr);
    if (n < 0) {
        if (errno == EPIPE) {
            bridge_fatal("Python side is gone (EPIPE on req_notify.fifo)");
        }
        bridge_fatal("write req_notify.fifo: " + std::string(std::strerror(errno)));
    }
}

GraphBatch FileDecisionBridge::deliver_and_receive(const StateDelta& delta) {
    // Phase 6 (方案 §9.1): bridge_ns -- the full blocking round trip from the
    // C++ side (write request + notify + wait for Python + read response).
    // Every exit before the final accounting is bridge_fatal (abort), so the
    // only counted exit is the successful return.
    const auto bridge_t0 = std::chrono::steady_clock::now();
    const uint64_t seq = delta.delivery_sequence;
    write_file_atomic(request_path(seq), build_request_json(delta));
    // Defect-B fix: ensure BOTH long-lived channel ends exist (lazy for
    // fixtures that never call open_notify(); the official entry calls it at
    // startup). The req write open below retries until Python's req reader
    // lands, so by the time notify_python() writes the doorbell Python's
    // serve loop is guaranteed running (its resp write end open follows its
    // req read end open -- decision_bridge.py).
    notify_python();
    // Wait for the response byte on the LIFETIME read end (defect-B fix):
    // no per-exchange open/close handshake anymore; read()==0 can only be
    // the peer's death (see wait_response_byte / open_notify).
    wait_response_byte(seq);

    nlohmann::json resp;
    try {
        resp = read_json_file(response_path(seq));
    } catch (const std::exception& exc) {
        bridge_fatal(std::string("response missing or corrupt for "
                                 "delivery_sequence=") +
                     std::to_string(seq) + ": " + exc.what());
    }
    // The three header field extractions run inside the same fail-closed
    // channel: value() throws a nlohmann type_error when a key is present
    // with an unexpected JSON type, and an uncaught throw would terminate
    // the process through the noexcept EventQueue::proceed() WITHOUT the
    // [Error] line the fail-closed contract promises. Absent keys keep
    // their defaults below; the check order (schema -> seq -> error) stays
    // frozen.
    int schema_version = -1;
    uint64_t source_seq = 0;
    std::string error;
    try {
        schema_version = resp.value("schema_version", -1);
        source_seq = resp.value("source_delivery_sequence", uint64_t(-1));
        error = resp.value("error", std::string());
    } catch (const std::exception& exc) {
        bridge_fatal(std::string("response field type violation for "
                                 "delivery_sequence=") +
                     std::to_string(seq) + ": " + exc.what());
    }
    if (schema_version != kDecisionBridgeSchemaVersion) {
        bridge_fatal("response schema_version mismatch for delivery_sequence=" +
                     std::to_string(seq));
    }
    if (source_seq != seq) {
        bridge_fatal("response source_delivery_sequence=" +
                     std::to_string(source_seq) + " != request seq=" +
                     std::to_string(seq));
    }

    // Error before structure (frozen ordering, rule T3/O4): a decision
    // failure aborts with the Python error message, never with a misleading
    // structural diagnostic -- the _fail skeleton carries all-empty arrays
    // and would parse cleanly anyway.
    if (!error.empty()) {
        bridge_fatal("Python decision failed for delivery_sequence=" +
                     std::to_string(seq) + ": " + error);
    }

    // C1 (2026-08-29): ONE structural parse into the typed ParsedGraphBatch.
    // The pre-C1 path moved the six DOM arrays into the batch and let
    // validate / liveness-preflight / commit assembly / anchor registration
    // re-extract every field (four full DOM walks per batch, each .value()
    // a std::map lookup + variant conversion + std::string deep copy). All
    // of that collapses into this single pass; the response DOM (resp) is
    // destroyed when this function returns. parse_graph_batch is a pure
    // local construction -- a ParseError unwinds only local vectors, so the
    // abort below leaves zero state side effects anywhere. The response
    // file also survives (post-mortem evidence; the unlink is only on the
    // success path).
    GraphBatch batch;
    try {
        batch = parse_graph_batch(resp, num_ranks_);
    } catch (const ParseError& exc) {
        bridge_fatal(std::string("malformed GraphBatch response for "
                                 "delivery_sequence=") +
                     std::to_string(seq) + ": " + exc.what());
    }
    // Phase 6 (方案 §9.1): round-trip accounting. B1 (2026-08-23): the
    // response contribution to channel_bytes is the on-disk response file
    // size (stat before the unlink), matching the request side's exact
    // body-bytes count.
    stats_.roundtrip_count += 1;
    stats_.roundtrip_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - bridge_t0)
            .count();
    // B1 (2026-08-23): the response's channel_bytes contribution is the
    // REAL on-disk file size (stat, taken before the unlink below). The
    // previous resp.dump().size() re-serialized the whole response just to
    // count bytes and reported the compact-dump length, which is NOT the
    // byte count that crossed the channel (the Python side writes spaced
    // json.dump files; the request side already counts exact body bytes at
    // write_file_atomic). Statistical field only (allowed-diff category):
    // the value shifts slightly upward to the true bridged bytes.
    struct stat response_stat {};
    if (::stat(response_path(seq).c_str(), &response_stat) != 0) {
        bridge_fatal("stat response " + response_path(seq) + ": " +
                     std::strerror(errno));
    }
    stats_.channel_bytes += static_cast<uint64_t>(response_stat.st_size);
    // Phase 7 (方案 §10.3): intermediate-product lifecycle. The response
    // file is fully consumed (the batch is materialized above); delete it
    // right away so the bridge dir holds only the request files (the
    // decision-sequence evidence replay-based re-checks consume) plus the
    // in-flight response. Measured on the frozen 20.csv first-30s input:
    // response files are ~202 MB of the ~252 MB bridge footprint. The
    // Python journals each successfully handled request into one ordered
    // request_journal.jsonl and retires loose request files in bounded batches;
    // idempotency/audit readers accept that journal and the legacy loose layout.
    // On fail-closed paths the response stays on disk for post-mortem.
    if (::unlink(response_path(seq).c_str()) != 0) {
        bridge_fatal("unlink response " + response_path(seq) + ": " +
                     std::strerror(errno));
    }
    return batch;
}

std::string FileDecisionBridge::stats_report() const {
    std::ostringstream os;
    os << "bridge_roundtrip_count=" << stats_.roundtrip_count
       << " bridge_ns=" << stats_.roundtrip_ns << " avg_bridge_ns=";
    if (stats_.roundtrip_count > 0) {
        os << stats_.roundtrip_ns / stats_.roundtrip_count;
    } else {
        os << "0";
    }
    os << " bridge_bytes=" << stats_.channel_bytes
       << " bridge_flush_count=" << stats_.forced_flush_count;
    return os.str();
}

void FileDecisionBridge::send_commit_ack(const uint64_t batch_id,
                                         const uint64_t delivery_seq,
                                         const bool success) {
    nlohmann::json ack;
    ack["schema_version"] = kDecisionBridgeSchemaVersion;
    ack["delivery_sequence"] = delivery_seq;
    ack["batch_id"] = batch_id;
    ack["success"] = success;
    write_file_atomic(ack_path(delivery_seq), ack);
    notify_python();
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
