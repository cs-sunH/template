/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

DecisionBridge -- execution-driven mechanism layer (wscllm phase 1).
Shm-implementation of the bridge protocol v1 (方案 §4 步骤 1-7; joint v11 §8
保底: the frame bytes that used to land in request_/response_/commit_ack_
files now travel over ShmFrameChannel POSIX-shm segments -- same bytes,
same FIFO doorbells, same fail-closed semantics, no loose frame files).

Fail-closed semantics: every protocol violation (Python crash via EOF,
EPIPE on the notification write, poll timeout, missing/corrupt response,
frame checksum mismatch, frame oversize, schema mismatch, non-empty error
field) prints one [Error] line and aborts. The static binary never
constructs a bridge (zero impact on the baseline).
*******************************************************************************/

#include "astra-sim/workload/execution_driven/DecisionBridge.hh"

#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
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
#include <set>
#include <vector>
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
        case DecisionReason::THERMAL_EPOCH:
            return "THERMAL_EPOCH";
    }
    // Dead-code cleanup (2026-09-29): the switch is exhaustive over
    // DecisionReason; the old trailing `return "UNKNOWN";` was unreachable
    // and existed only to pacify -Wreturn-type. An out-of-enum value is a
    // programming error: fail loud instead of printing a bogus reason name
    // into the request JSON (same idiom as bridge_fatal).
    std::fprintf(stderr,
                 "[Error] (execution_driven/bridge) reason_name: unreachable "
                 "reason value %d\n",
                 static_cast<int>(reason));
    std::abort();
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

nlohmann::json parse_json_strict(const std::string& payload,
                                 const std::string& what) {
    // Strict JSON load (same rules as the former read_json_file): the
    // istream operator>> parses in NON-strict mode -- it silently consumes
    // the FIRST JSON value and leaves trailing bytes unread, and the DOM
    // build last-wins on duplicate object keys. These payloads are
    // machine-written by paired tooling, so trailing content or a
    // duplicate key is producer corruption, not a legal shape.
    // json::parse in its default strict mode rejects trailing bytes
    // outright; the stateful parse callback rejects a repeated key inside
    // one object.
    struct KeyGuard {
        std::vector<std::set<std::string>> stack;
        std::string duplicate;
        bool operator()(int, nlohmann::json::parse_event_t event,
                        nlohmann::json& parsed) {
            using EVT = nlohmann::json::parse_event_t;
            if (event == EVT::object_start) {
                stack.emplace_back();
            } else if (event == EVT::key) {
                if (!stack.back().insert(parsed.get<std::string>()).second) {
                    duplicate = parsed.get<std::string>();
                }
            } else if (event == EVT::object_end && !stack.empty()) {
                stack.pop_back();
            }
            return true;  // observe only; never discard subtree
        }
    } guard;
    nlohmann::json value;
    try {
        value = nlohmann::json::parse(payload, guard);
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("corrupt JSON in " + what + ": " + e.what());
    }
    if (!guard.duplicate.empty()) {
        throw std::runtime_error("duplicate JSON key \"" + guard.duplicate +
                                 "\" in " + what);
    }
    return value;
}

// poll() helper: 0 = timeout, 1 = readable (POLLIN or POLLHUP), -1 = error.
// timeout_ms == 0 means wait forever. L9 (2026-09-29): EINTR retries the
// poll instead of surfacing as a fatal channel error -- a signal
// interruption is not a bridge failure (the timeout restarts on the retry;
// the bound stays a liveness bound, not an exact deadline).
int wait_readable(int fd, int timeout_ms) {
    struct pollfd pfd;
    pfd.fd = fd;
    pfd.events = POLLIN | POLLHUP;
    pfd.revents = 0;
    const int ms = (timeout_ms == 0) ? -1 : timeout_ms;
    int r;
    do {
        r = ::poll(&pfd, 1, ms);
    } while (r < 0 && errno == EINTR);
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
    // W1 (joint work ledger): the mailbox-local event sequence, additive and
    // parallel to reasons[] (index i of event_seqs is the seq of the event
    // reasons[i]). Preserves the intra-epoch event order on the wire: the
    // mailbox assigns push-order seqs at drain time, so (delivery_sequence,
    // event_seq) is a stable identity for the epoch's decision events. The
    // reasons[] order itself was already frozen (mailbox drain order), so
    // this array is a pure additive disclosure -- old peers that ignore it
    // parse identically.
    req["event_seqs"] = nlohmann::json::array();
    req["arrivals"] = nlohmann::json::array();
    req["completed_groups"] = nlohmann::json::array();
    // Phase 4 (v1): per-node terminal facts since the previous delivery.
    req["completed_nodes"] = nlohmann::json::array();
    for (const auto& fact : delta.completed_nodes) {
        nlohmann::json f;
        f["rank"] = fact.rank;
        f["node_id"] = fact.node_id;
        f["request_id"] = fact.request_id;
        f["stage"] = fact.stage;
        f["generation"] = fact.generation;
        f["tick"] = fact.tick;
        f["terminal_status"] = fact.terminal_status;
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
        req["event_seqs"].push_back(ev.seq);
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
            case DecisionReason::THERMAL_EPOCH:
                // Plan D: the request-independent periodic thermal epoch. It
                // carries no request/group payload -- the reason name alone is
                // the wire fact (arrivals[]/completed_groups[] stay empty). The
                // thermal facts travel as the additive top-level
                // thermal_activity sibling (ThermalActivityProvider), not
                // inside this per-event loop.
                break;
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
    if (!delta.workspace_state.empty()) {
        nlohmann::json ranks = nlohmann::json::array();
        for (const auto& state : delta.workspace_state) {
            ranks.push_back({{"rank", state.rank},
                             {"workspace_revision",
                              state.workspace_revision},
                             {"active_workspace_bytes",
                              state.active_workspace_bytes}});
        }
        req["workspace_state"] = {
            {"protocol_version", 1}, {"ranks", std::move(ranks)}};
    }
    return req;
}

// -------------------------------------------------------------- ShmBridge --

namespace {

// The C++ -> Python segment pointer written into <bridge_dir>. Python
// (online/bridge_shm.py) loads it fail-closed at serve start and validates
// the generation -- the cross-run pollution guard.
constexpr const char* kBridgeShmInfoName = "bridge_shm.json";

nlohmann::json load_bridge_shm_info_json(const std::string& bridge_dir) {
    const std::string path = bridge_dir + "/" + kBridgeShmInfoName;
    std::ifstream in(path);
    if (!in) {
        return nlohmann::json();
    }
    try {
        return nlohmann::json::parse(in);
    } catch (const nlohmann::json::exception&) {
        return nlohmann::json();
    }
}

}  // namespace

ShmDecisionBridge::ShmDecisionBridge(std::string bridge_dir,
                                     const int timeout_ms,
                                     const int num_ranks)
    : bridge_dir_(std::move(bridge_dir)),
      timeout_ms_(timeout_ms),
      num_ranks_(num_ranks) {
    // v11 §8: the carrier is prepared at construction (ensure_bridge_dir
    // already did this in-process for the official path; this call adopts
    // the existing segments so no second generation is minted between the
    // runner's FIFO-ready poll and Python's bridge_shm.json read).
    ensure_bridge_shm();
}

ShmDecisionBridge::~ShmDecisionBridge() {
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
    if (shm_prepared_) {
        // Unmap first, then unlink the names. A Python still finishing its
        // exit keeps its own mappings (unlink never invalidates maps), and
        // the next run's ensure_bridge_dir mints a fresh generation anyway.
        req_channel_.close();
        ack_channel_.close();
        resp_channel_.close();
        const char* names[] = {shm_request_segment_.c_str(),
                               shm_commit_ack_segment_.c_str(),
                               shm_response_segment_.c_str()};
        for (const char* name : names) {
            std::string slash = std::string("/") + name;
            if (::shm_unlink(slash.c_str()) != 0 && errno != ENOENT) {
                // Best-effort teardown: never throw out of a destructor.
                // A leaked segment is reclaimed by the next run's
                // clean_stale_bridge_shm.
                std::fprintf(stderr,
                             "[Error] (execution_driven/bridge) destructor "
                             "shm_unlink %s: %s\n",
                             name, std::strerror(errno));
            }
        }
        // bridge_shm.json is deliberately LEFT in place: it is this run's
        // post-mortem pointer (generation + segment names) and is
        // atomically rewritten by the next ensure_bridge_dir.
        shm_prepared_ = false;
    }
}

void ShmDecisionBridge::ensure_bridge_dir(const std::string& bridge_dir) {
    mkdir_p(bridge_dir);
    ensure_fifo(bridge_dir + "/req_notify.fifo");
    ensure_fifo(bridge_dir + "/resp_notify.fifo");
    // Clean stale loose frame files from a previous run in the same
    // directory (file-transport era hygiene; also .tmp leftovers).
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
    // Mint (or leave alone, when already minted in this process) the shm
    // carrier BEFORE either side starts: Python loads bridge_shm.json at
    // serve start, so the segments and the pointer file must exist by then
    // (the runner gates Python on the FIFOs, which are created above).
    mint_bridge_shm_info(bridge_dir);
}

// static
void ShmDecisionBridge::mint_bridge_shm_info(const std::string& bridge_dir) {
    // Adopt path: a valid pointer file whose segments exist and validate
    // (minted earlier in this process by ensure_bridge_dir). Everything
    // else -- missing info, dangling names, corrupt header -- mints a
    // fresh generation after cleaning this bridge dir's stale segments.
    // The channels held here are CLOSED before returning (close only
    // unmaps; segment unlinking is exclusively the bridge instance
    // destructor's job).
    const nlohmann::json info = load_bridge_shm_info_json(bridge_dir);
    bool adopted = false;
    if (info.is_object() && info.value("version", 0) == 1) {
        try {
            const uint64_t generation =
                info.at("generation").get<uint64_t>();
            const uint64_t capacity =
                info.at("slot_capacity").get<uint64_t>();
            const std::string req =
                info.at("request_segment").get<std::string>();
            const std::string ack =
                info.at("commit_ack_segment").get<std::string>();
            const std::string resp =
                info.at("response_segment").get<std::string>();
            if (capacity == kBridgeShmSlotCapacity) {
                ShmFrameChannel c0 = ShmFrameChannel::attach(
                    std::string("/") + req, generation, capacity);
                ShmAckRing c1 = ShmAckRing::attach(
                    std::string("/") + ack, generation,
                    kBridgeShmAckSlotCount, kBridgeShmAckSlotBytes);
                ShmFrameChannel c2 = ShmFrameChannel::attach(
                    std::string("/") + resp, generation, capacity);
                (void)c0;
                (void)c1;
                (void)c2;
                adopted = true;
            }
        } catch (const std::exception&) {
            adopted = false;
        }
    }
    if (adopted) {
        return;
    }
    clean_stale_bridge_shm(bridge_dir);
    const uint64_t generation = new_bridge_shm_generation();
    const std::string stem = bridge_shm_name_stem(bridge_dir);
    char gen_hex[32];
    std::snprintf(gen_hex, sizeof(gen_hex), "%016llx",
                  static_cast<unsigned long long>(generation));
    const std::string req = stem + "req_" + gen_hex;
    const std::string ack = stem + "ack_" + gen_hex;
    const std::string resp = stem + "resp_" + gen_hex;
    ShmFrameChannel c0 = ShmFrameChannel::create(
        std::string("/") + req, generation, kBridgeShmSlotCapacity);
    ShmAckRing c1 = ShmAckRing::create(
        std::string("/") + ack, generation, kBridgeShmAckSlotCount,
        kBridgeShmAckSlotBytes);
    ShmFrameChannel c2 = ShmFrameChannel::create(
        std::string("/") + resp, generation, kBridgeShmSlotCapacity);
    (void)c0;
    (void)c1;
    (void)c2;
    // Pointer file for Python: atomic publish (tmp + rename), same
    // discipline the old transport's frame files used.
    nlohmann::json out;
    out["version"] = 1;
    out["generation"] = generation;
    out["request_segment"] = req;
    out["commit_ack_segment"] = ack;
    out["response_segment"] = resp;
    out["slot_capacity"] = kBridgeShmSlotCapacity;
    const std::string path = bridge_dir + "/" + kBridgeShmInfoName;
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp);
        if (!f) {
            bridge_fatal("cannot open " + tmp);
        }
        f << out.dump();
        f.flush();
        if (!f) {
            bridge_fatal("write failed for " + tmp);
        }
    }
    if (::rename(tmp.c_str(), path.c_str()) != 0) {
        bridge_fatal("rename " + tmp + " -> " + path + ": " +
                     std::strerror(errno));
    }
}

void ShmDecisionBridge::ensure_bridge_shm() {
    if (shm_prepared_) {
        return;
    }
    mint_bridge_shm_info(bridge_dir_);
    const nlohmann::json info = load_bridge_shm_info_json(bridge_dir_);
    if (!info.is_object() || info.value("version", 0) != 1) {
        bridge_fatal("bridge shm info missing or invalid after mint (" +
                     bridge_dir_ + "/" + kBridgeShmInfoName + ")");
    }
    try {
        const uint64_t generation = info.at("generation").get<uint64_t>();
        const uint64_t capacity = info.at("slot_capacity").get<uint64_t>();
        shm_request_segment_ = info.at("request_segment").get<std::string>();
        shm_commit_ack_segment_ =
            info.at("commit_ack_segment").get<std::string>();
        shm_response_segment_ =
            info.at("response_segment").get<std::string>();
        req_channel_ = ShmFrameChannel::attach(
            std::string("/") + shm_request_segment_, generation, capacity);
        ack_channel_ = ShmAckRing::attach(
            std::string("/") + shm_commit_ack_segment_, generation,
            kBridgeShmAckSlotCount, kBridgeShmAckSlotBytes);
        resp_channel_ = ShmFrameChannel::attach(
            std::string("/") + shm_response_segment_, generation, capacity);
        shm_generation_ = generation;
        shm_slot_capacity_ = capacity;
    } catch (const std::exception& exc) {
        bridge_fatal("cannot attach the bridge shm segments: " +
                     std::string(exc.what()));
    }
    shm_prepared_ = true;
}

void ShmDecisionBridge::publish_request_frame(const uint64_t seq,
                                              const std::string& body) {
    // Channel-byte and forced-flush accounting (方案 §9.1 / §15): the frame
    // payload is exactly the bytes the old atomic file write flushed, so
    // the counters keep their meaning -- one flush per published frame,
    // O(deliveries), never O(nodes).
    stats_.channel_bytes += body.size();
    stats_.forced_flush_count += 1;
    try {
        req_channel_.publish(seq, body);
    } catch (const std::exception& exc) {
        bridge_fatal(std::string("request frame publish failed for "
                                 "delivery_sequence=") +
                     std::to_string(seq) + ": " + exc.what());
    }
}

void ShmDecisionBridge::open_notify() {
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

void ShmDecisionBridge::wait_response_byte(const uint64_t seq) {
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
            return;  // the notification byte (the response frame is in shm)
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
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
            continue;  // spurious wake or signal interruption: re-enter poll
        }
        bridge_fatal("read resp_notify.fifo: " +
                     std::string(std::strerror(errno)));
    }
}

void ShmDecisionBridge::notify_python() {
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
    // L9 (2026-09-29): EINTR/EAGAIN retry the 1-byte doorbell write instead
    // of failing the bridge -- a signal interruption (or a transiently full
    // FIFO buffer under the 1:1 backpressure contract) is retryable, not a
    // dead peer. The channel is one byte per round trip, so the retry
    // cannot spin in practice.
    ssize_t n;
    do {
        n = ::write(req_notify_fd_, &byte, 1);
    } while (n < 0 &&
             (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK));
    ::sigaction(SIGPIPE, &old_action, nullptr);
    if (n < 0) {
        if (errno == EPIPE) {
            bridge_fatal("Python side is gone (EPIPE on req_notify.fifo)");
        }
        bridge_fatal("write req_notify.fifo: " + std::string(std::strerror(errno)));
    }
}

void ShmDecisionBridge::set_link_telemetry_provider(
    LinkTelemetryProvider provider) {
    // C6 (WP2, --link-telemetry): startup-wiring contract -- the online
    // main installs the provider before the event loop, so a replacement
    // after deliveries began would silently change the request schema
    // mid-run. Fail closed instead of accepting the swap.
    if (stats_.roundtrip_count > 0) {
        bridge_fatal("set_link_telemetry_provider after the first delivery "
                     "(the request schema cannot change mid-run)");
    }
    link_telemetry_provider_ = std::move(provider);
}

void ShmDecisionBridge::set_work_ledger_provider(
    WorkLedgerProvider provider) {
    // W1: same startup-only contract as the telemetry provider.
    if (stats_.roundtrip_count > 0) {
        bridge_fatal("set_work_ledger_provider after the first delivery "
                     "(the request schema cannot change mid-run)");
    }
    work_ledger_provider_ = std::move(provider);
}

void ShmDecisionBridge::set_thermal_activity_provider(
    ThermalActivityProvider provider) {
    // Plan D: same startup-only contract as the telemetry/ledger providers.
    // The feature flag (SH_THERMAL_EPOCH_NS > 0) is read once at startup, so a
    // mid-run swap would silently change the request schema between epochs of
    // one run. Fail closed instead of accepting the swap.
    if (stats_.roundtrip_count > 0) {
        bridge_fatal("set_thermal_activity_provider after the first delivery "
                     "(the request schema cannot change mid-run)");
    }
    thermal_activity_provider_ = std::move(provider);
}

void ShmDecisionBridge::set_thermal_cap_consumer(
    ThermalCapConsumer consumer) {
    // Plan D: startup-only wiring (the consumer owns the coarse runtimes).
    if (stats_.roundtrip_count > 0) {
        bridge_fatal("set_thermal_cap_consumer after the first delivery "
                     "(the response consumer cannot change mid-run)");
    }
    thermal_cap_consumer_ = std::move(consumer);
}

// ------------------------------------------------- W1 work-ledger diff --

namespace {

// One section's entries flattened to (work_id -> entry copy). Copies, not
// pointers: the blocks are read out of temporaries (state.value(...) returns
// by value), so a pointer map would dangle at the end of each loop
// iteration.
using WorkKeyedMap = std::map<std::string, nlohmann::json>;

WorkKeyedMap index_section(const nlohmann::json& array) {
    WorkKeyedMap indexed;
    if (array.is_array()) {
        for (const auto& entry : array) {
            if (entry.is_object() && entry.contains("work_id")) {
                indexed.emplace(entry.at("work_id").get<std::string>(),
                                entry);
            }
        }
    }
    return indexed;
}

// Flatten the four sections of a state frame. Sections without per-key
// identity (noc links, slot counters) are intentionally NOT diffed -- they
// are gauge-style facts re-exported with every full frame, not lifecycle
// works.
struct StateIndex {
    WorkKeyedMap nodes;
    WorkKeyedMap hbm;
    WorkKeyedMap pool;
    WorkKeyedMap flows;
};

StateIndex index_state(const nlohmann::json& state) {
    StateIndex index;
    if (!state.is_object()) {
        return index;
    }
    // nodes / hbm: per-rank blocks; pool: per-port blocks.
    for (const char* const section : {"nodes", "hbm", "pool"}) {
        const auto section_it = state.find(section);
        if (section_it == state.end() || !section_it->is_array()) {
            continue;
        }
        for (const auto& block : *section_it) {
            for (const char* const key : {"nodes", "jobs"}) {
                const auto it = block.find(key);
                if (it == block.end()) {
                    continue;
                }
                for (const auto& entry : *it) {
                    if (entry.is_object() && entry.contains("work_id")) {
                        WorkKeyedMap& target =
                            section[0] == 'n' ? index.nodes
                            : section[0] == 'h' ? index.hbm
                                                : index.pool;
                        target.emplace(entry.at("work_id").get<std::string>(),
                                       entry);
                    }
                }
            }
        }
    }
    const auto noc = state.find("noc");
    if (noc != state.end() && noc->is_object()) {
        const auto flows = noc->find("flows");
        if (flows != noc->end()) {
            index.flows = index_section(*flows);
        }
    }
    return index;
}

nlohmann::json make_event_id(const uint64_t delivery_sequence,
                             const uint64_t tick, uint64_t& next_seq) {
    return nlohmann::json{
        {"tick", tick},
        {"delivery_sequence", delivery_sequence},
        {"producer", "work_ledger_diff"},
        {"local_seq", next_seq++},
    };
}

nlohmann::json make_event(const std::string& kind, const std::string& work_id,
                          const uint64_t tick, const uint64_t delivery_sequence,
                          uint64_t& next_seq, nlohmann::json extra) {
    nlohmann::json event = nlohmann::json{
        {"event_id", make_event_id(delivery_sequence, tick, next_seq)},
        {"kind", kind},
        {"work_id", work_id},
        {"tick", tick},
    };
    event.merge_patch(std::move(extra));
    return event;
}

// Stable identity facts plus the authoritative residual a first-seen
// resource leg carries (the Python ledger builds its WorkIdentity and
// remaining state from these fields).
nlohmann::json commit_payload(const nlohmann::json& entry) {
    nlohmann::json payload;
    // NOTE: "kind" (the node kind name) is deliberately NOT carried -- the
    // event-level "kind" discriminator is the lifecycle kind and must not
    // be shadowed by the payload merge; the operation name travels as
    // "operation".
    static const char* const kCarry[] = {
        "request_id", "stage", "generation", "rank", "node_id", "operation",
        "leg", "num_ops", "bytes", "runtime_ns", "tensor_size",
        "total_bytes", "dependencies", "parents", "service_unit", "route",
        "flow_id", "port_index", "issue_seq", "state",
        // Resource legs may first enter the delta after they have already
        // received service.  Keep the backend's authoritative residual in
        // the commit itself; carrying only the static total silently turns
        // a mid-flight first sighting into a stale upper bound on Python.
        "remaining_ops", "remaining_bytes", "remaining_latency_ns",
        "total_latency_ns"};
    for (const char* key : kCarry) {
        const auto it = entry.find(key);
        if (it != entry.end()) {
            payload[key] = *it;
        }
    }
    nlohmann::json total = nlohmann::json::object();
    if (entry.contains("num_ops")) {
        total["ops"] = entry.at("num_ops");
    }
    if (entry.contains("bytes")) {
        total["bytes"] = entry.at("bytes");
    } else if (entry.contains("total_bytes")) {
        total["bytes"] = entry.at("total_bytes");
    }
    if (entry.contains("total_latency_ns")) {
        total["latency_ns"] = entry.at("total_latency_ns");
    } else if (entry.contains("runtime_ns")) {
        total["latency_ns"] = entry.at("runtime_ns");
    }
    if (!total.empty()) {
        payload["total"] = std::move(total);
    }
    return payload;
}

// One section's diff. `compact_out` non-null selects the compact node
// encoding (see DecisionBridge.hh); lifecycle changes land in the four
// lists and only progress events keep the full envelope.
struct CompactOut {
    nlohmann::json* commits = nullptr;
    nlohmann::json* issues = nullptr;
    nlohmann::json* completions = nullptr;
    nlohmann::json* vanished = nullptr;
};

void diff_one_section(const WorkKeyedMap& prev, const WorkKeyedMap& cur,
                      const uint64_t tick, const uint64_t delivery_sequence,
                      uint64_t& next_seq, nlohmann::json& events,
                      const CompactOut& compact_out = CompactOut{}) {
    static const char* const kProgressFields[] = {
        "remaining_ops", "remaining_bytes", "remaining_latency_ns"};
    const bool compact = compact_out.commits != nullptr;
    for (const auto& cur_item : cur) {
        const std::string& work_id = cur_item.first;
        const nlohmann::json& cur_entry = cur_item.second;
        const auto prev_it = prev.find(work_id);
        if (prev_it == prev.end()) {
            // New work: commit, then replay the current lifecycle flags.
            if (compact) {
                compact_out.commits->push_back(cur_entry);
                ++next_seq;
                continue;
            }
            nlohmann::json payload = commit_payload(cur_entry);
            const bool issued = cur_entry.value("issued", false);
            const bool finished = cur_entry.value("finished", false);
            if (!payload.contains("state")) {
                payload["state"] = finished ? "completed"
                                             : (issued ? "issued"
                                                       : "committed");
            }
            events.push_back(make_event("commit", work_id, tick,
                                        delivery_sequence, next_seq,
                                        std::move(payload)));
            continue;
        }
        const nlohmann::json& prev_entry = prev_it->second;
        if (cur_entry.value("issued", false) &&
            !prev_entry.value("issued", false)) {
            if (compact) {
                compact_out.issues->push_back(work_id);
                ++next_seq;
            } else {
                events.push_back(make_event("issue", work_id, tick,
                                            delivery_sequence, next_seq,
                                            nlohmann::json::object()));
            }
        }
        for (const char* field : kProgressFields) {
            const auto cur_it = cur_entry.find(field);
            const auto prev_it2 = prev_entry.find(field);
            if (cur_it == cur_entry.end() || prev_it2 == prev_entry.end()) {
                continue;
            }
            const double served =
                prev_it2->get<double>() - cur_it->get<double>();
            if (served > 0.5e-9) {
                const std::string dim =
                    std::string(field) == "remaining_ops" ? "ops"
                    : std::string(field) == "remaining_bytes" ? "bytes"
                                                              : "latency_ns";
                // Keep the producer's current residual beside the derived
                // delta.  A consumer can reconcile the subtraction against
                // the backend's tick-end value and avoid turning a rounded
                // diff into a silently different ledger remainder.
                nlohmann::json progress_payload = {
                    {"delta_served", nlohmann::json{{dim, served}}}};
                progress_payload[field] = *cur_it;
                events.push_back(make_event(
                    "progress", work_id, tick, delivery_sequence, next_seq,
                    std::move(progress_payload)));
            }
        }
        if (cur_entry.value("finished", false) &&
            !prev_entry.value("finished", false)) {
            if (compact) {
                compact_out.completions->push_back(work_id);
                ++next_seq;
            } else {
                events.push_back(make_event("completion", work_id, tick,
                                            delivery_sequence, next_seq,
                                            nlohmann::json::object()));
            }
        }
    }
    for (const auto& prev_item : prev) {
        const std::string& work_id = prev_item.first;
        const nlohmann::json& prev_entry = prev_item.second;
        if (cur.find(work_id) != cur.end()) {
            continue;
        }
        // Vanished: NodeStore GC collected a finished node / the resource
        // model completed a job. Emit the missing terminal transitions.
        if (compact) {
            compact_out.vanished->push_back(work_id);
            // The Python side deterministically expands every vanished id
            // into completion + release (completion idempotent on an
            // already-completed work), so the counter consumes exactly 2.
            next_seq += 2;
            continue;
        }
        if (!prev_entry.value("finished", false)) {
            events.push_back(make_event("completion", work_id, tick,
                                        delivery_sequence, next_seq,
                                        nlohmann::json::object()));
        }
        events.push_back(make_event("ack", work_id, tick, delivery_sequence,
                                    next_seq, nlohmann::json::object()));
        events.push_back(make_event("release", work_id, tick,
                                    delivery_sequence, next_seq,
                                    nlohmann::json::object()));
    }
}

}  // namespace

WorkLedgerDiff diff_work_ledger_states(const nlohmann::json& prev_state,
                                       const nlohmann::json& cur_state,
                                       const uint64_t delivery_sequence,
                                       const uint64_t tick,
                                       uint64_t& next_local_seq) {
    const StateIndex prev = index_state(prev_state);
    const StateIndex cur = index_state(cur_state);
    WorkLedgerDiff result;
    result.events = nlohmann::json::array();
    nlohmann::json node_commits = nlohmann::json::array();
    nlohmann::json node_issues = nlohmann::json::array();
    nlohmann::json node_completions = nlohmann::json::array();
    nlohmann::json node_vanished = nlohmann::json::array();
    CompactOut compact_out{&node_commits, &node_issues, &node_completions,
                           &node_vanished};
    diff_one_section(prev.nodes, cur.nodes, tick, delivery_sequence,
                     next_local_seq, result.events, compact_out);
    diff_one_section(prev.hbm, cur.hbm, tick, delivery_sequence,
                     next_local_seq, result.events);
    diff_one_section(prev.pool, cur.pool, tick, delivery_sequence,
                     next_local_seq, result.events);
    diff_one_section(prev.flows, cur.flows, tick, delivery_sequence,
                     next_local_seq, result.events);
    const bool any_compact = !node_commits.empty() || !node_issues.empty() ||
                             !node_completions.empty() ||
                             !node_vanished.empty();
    if (any_compact) {
        const uint64_t total_compact = node_commits.size() +
                                       node_issues.size() +
                                       node_completions.size() +
                                       2 * node_vanished.size();
        // The nodes section runs FIRST, so the compact entries consumed
        // seqs [base, base+total_compact) and the leg/progress events
        // consumed [base+total_compact, counter): subtract BOTH tails.
        result.compact = nlohmann::json{
            {"node_commits", std::move(node_commits)},
            {"node_issues", std::move(node_issues)},
            {"node_completions", std::move(node_completions)},
            {"node_vanished", std::move(node_vanished)},
            {"seq_base", next_local_seq - total_compact -
                             result.events.size()},
        };
    } else {
        result.compact = nlohmann::json();
    }
    return result;
}

GraphBatch ShmDecisionBridge::deliver_and_receive(const StateDelta& delta) {
    // Phase 6 (方案 §9.1): bridge_ns -- the full blocking round trip from the
    // C++ side (publish request frame + notify + wait for Python + read
    // response frame). Every exit before the final accounting is
    // bridge_fatal (abort), so the only counted exit is the successful
    // return.
    const auto bridge_t0 = std::chrono::steady_clock::now();
    const uint64_t seq = delta.delivery_sequence;
    // C6 (WP2, --link-telemetry): attach the optional top-level
    // link_telemetry[] array as a sibling of ledger_summary (additive
    // field; the v0 snapshot reserved object is untouched). Unset provider
    // (flag off, the fail-closed default) = the field is omitted and the
    // serialized request bytes are identical to the pre-C6 wire format.
    // The provider samples the fluid observer's cumulative totals exactly
    // once per delivery epoch (1:1 backpressure above).
    nlohmann::json request = build_request_json(delta);
    if (link_telemetry_provider_) {
        request["link_telemetry"] = link_telemetry_provider_(delta);
    }
    // Plan D (periodic thermal epoch): attach the optional top-level
    // thermal_activity sibling (additive; absent when the feature is off, so
    // the request bytes stay byte-identical to the pre-D wire format). The
    // provider samples the coarse die runtime's cumulative served counters
    // exactly once per delivery epoch (1:1 backpressure above). The flag is
    // captured once: it is EXACTLY the condition under which the response may
    // legally carry thermal_cap (rule T6 conditional allow, see
    // ParsedGraphBatch), so the request and the parse must agree on it.
    const bool request_has_thermal_activity =
        static_cast<bool>(thermal_activity_provider_);
    if (request_has_thermal_activity) {
        request["thermal_activity"] = thermal_activity_provider_(delta);
    }
    // W1/v6 (joint work ledger): splice the compact work_ledger frame into
    // the request when the provider is installed (sibling of link_telemetry
    // and ledger_summary; never occupies the v0 snapshot reserved slot).
    // The provider runs exactly once per delivery epoch (1:1 backpressure)
    // and returns the deterministic workledger-compact-v1 bytes. The splice
    // still happens HERE, before the frame is published, so the payload
    // bytes are byte-identical to the former write_request_with_ledger
    // file content: the raw binary member sits in place of the null
    // placeholder and the Python receiver's structural scan still removes
    // it exactly as it did from the file bytes.
    std::string body;
    if (work_ledger_provider_) {
        const WorkLedgerWirePayload wire = work_ledger_provider_(delta);
        request["work_ledger"] = nullptr;   // splice placeholder
        const std::string dumped = request.dump();
        // nlohmann's object keys are sorted and the member name is unique,
        // so the null placeholder surfaces as exactly one
        // "work_ledger":null fragment in the deterministic dump.
        static const std::string marker = "\"work_ledger\":null";
        const std::size_t at = dumped.find(marker);
        if (at == std::string::npos) {
            bridge_fatal("request payload lacks the work_ledger placeholder");
        }
        if (dumped.find(marker, at + 1) != std::string::npos) {
            bridge_fatal("request payload carries duplicate work_ledger keys");
        }
        body.reserve(dumped.size() + wire.bytes.size());
        body.append(dumped, 0, at + marker.size() - 4);  // up to ':' inclusive
        body.append(wire.bytes);
        body.append(dumped, at + marker.size(), std::string::npos);
    } else {
        body = request.dump();
    }
    publish_request_frame(seq, body);
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

    std::string resp_payload;
    uint64_t resp_seq = 0;
    try {
        if (!resp_channel_.try_read(seq, &resp_payload, &resp_seq)) {
            bridge_fatal("response frame missing or stale for "
                         "delivery_sequence=" + std::to_string(seq) +
                         " (empty or already-consumed slot after the "
                         "response byte)");
        }
        if (resp_seq != seq) {
            bridge_fatal("response frame seq " + std::to_string(resp_seq) +
                         " != request seq " + std::to_string(seq));
        }
    } catch (const std::exception& exc) {
        bridge_fatal(std::string("response frame corrupt for "
                                 "delivery_sequence=") +
                     std::to_string(seq) + ": " + exc.what());
    }
    nlohmann::json resp;
    try {
        resp = parse_json_strict(resp_payload,
                                 "response frame (delivery_sequence=" +
                                     std::to_string(seq) + ")");
    } catch (const std::exception& exc) {
        bridge_fatal(std::string("response missing or corrupt for "
                                 "delivery_sequence=") +
                     std::to_string(seq) + ": " + exc.what());
    }
    // (2026-10-01 审查修复) 三处字段读取显式类型判定：nlohmann value()
    // 对"键在场但类型不符"抛 type_error.302（非返回缺省）——此三处在
    // parse_graph_batch 的 try/catch 之前执行，裸异常击穿本文件"每个
    // 协议违反打印一行 [Error] 再 abort"的通道契约（无诊断 terminate）。
    if (!resp.contains("schema_version") ||
        !resp["schema_version"].is_number_integer()) {
        bridge_fatal("response schema_version missing or not an integer "
                     "for delivery_sequence=" + std::to_string(seq));
    }
    if (resp["schema_version"].get<int>() != kDecisionBridgeSchemaVersion) {
        bridge_fatal("response schema_version mismatch for delivery_sequence=" +
                     std::to_string(seq));
    }
    if (!resp.contains("source_delivery_sequence") ||
        !resp["source_delivery_sequence"].is_number_unsigned()) {
        bridge_fatal("response source_delivery_sequence missing or not an "
                     "unsigned integer for delivery_sequence=" +
                     std::to_string(seq));
    }
    const uint64_t source_seq =
        resp["source_delivery_sequence"].get<uint64_t>();
    if (source_seq != seq) {
        bridge_fatal("response source_delivery_sequence=" +
                     std::to_string(source_seq) + " != request seq=" +
                     std::to_string(seq));
    }

    // Error before structure (frozen ordering, rule T3/O4): a decision
    // failure aborts with the Python error message, never with a misleading
    // structural diagnostic -- the _fail skeleton carries all-empty arrays
    // and would parse cleanly anyway.
    if (resp.contains("error") && !resp["error"].is_string()) {
        bridge_fatal("response error field is not a string for "
                     "delivery_sequence=" + std::to_string(seq));
    }
    const std::string error =
        resp.contains("error") ? resp["error"].get<std::string>() : "";
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
    // abort below leaves zero state side effects anywhere.
    GraphBatch batch;
    try {
        batch = parse_graph_batch(resp, num_ranks_, delta.workspace_state,
                                  request_has_thermal_activity);
    } catch (const ParseError& exc) {
        bridge_fatal(std::string("malformed GraphBatch response for "
                                 "delivery_sequence=") +
                     std::to_string(seq) + ": " + exc.what());
    }
    // Plan D (periodic thermal epoch): hand the response's optional thermal_cap
    // to the installed consumer (the online main applies it to the coarse die
    // runtimes). Absent cap = the Python peer produced no throttling decision
    // this epoch (feature not yet wired, or the policy chose not to change the
    // cap): the bridge delivers nothing and continues -- absence is NEVER a
    // protocol violation (fail-open on absence is the frozen rule for an
    // additive response field). The consumer is optional; an uninstalled one
    // simply drops a delivered cap.
    if (batch.has_thermal_cap && thermal_cap_consumer_) {
        thermal_cap_consumer_(batch.thermal_cap);
    }
    // Phase 6 (方案 §9.1): round-trip accounting.
    stats_.roundtrip_count += 1;
    stats_.roundtrip_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - bridge_t0)
            .count();
    // B1 (2026-08-23) accounting, v11 §8 form: the response's
    // channel_bytes contribution is the frame payload length -- exactly
    // the bytes Python published, which under the old transport were the
    // response file's on-disk size (stat before the unlink). The counter's
    // meaning ("real bridged bytes", not a re-serialized dump length) is
    // unchanged.
    stats_.channel_bytes += static_cast<uint64_t>(resp_payload.size());
    // Phase 7 (方案 §10.3): intermediate-product lifecycle. The response
    // frame is fully consumed (the batch is materialized above); nothing is
    // deleted -- the single-slot channel is simply re-published by the next
    // epoch, and the bridge dir keeps only the journal/sidecars. On
    // fail-closed paths the frame stays parked in the shm slot; abort()
    // runs no destructors, so the segments survive in /dev/shm with the
    // offending frame as the post-mortem evidence (bridge_shm.json names
    // them).
    return batch;
}

std::string ShmDecisionBridge::stats_report() const {
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

void ShmDecisionBridge::send_commit_ack(const uint64_t batch_id,
                                        const uint64_t delivery_seq,
                                        const bool success) {
    nlohmann::json ack;
    ack["schema_version"] = kDecisionBridgeSchemaVersion;
    ack["delivery_sequence"] = delivery_seq;
    ack["batch_id"] = batch_id;
    ack["success"] = success;
    // Same accounting as the request path: one published frame = one
    // counted flush; the payload bytes are exactly the old
    // commit_ack_<seq>.json file content.
    const std::string body = ack.dump();
    stats_.channel_bytes += body.size();
    stats_.forced_flush_count += 1;
    try {
        ack_channel_.publish(delivery_seq, body);
    } catch (const std::exception& exc) {
        bridge_fatal(std::string("commit-ack frame publish failed for "
                                 "delivery_sequence=") +
                     std::to_string(delivery_seq) + ": " + exc.what());
    }
    notify_python();
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
