/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

DecisionBridge -- execution-driven mechanism layer (wscllm phase 1).
Interface + File implementation (方案 §4 步骤 1-7).

The C++ <-> Python decision channel of the Execution-Driven loop. The
tick-end gate (ed_driver_tick_end) drains the DecisionMailbox into one
StateDelta and calls deliver_and_receive(delta); the returned GraphBatch is
the scheduler's reply. This bridge is a DECISION channel, NOT a runtime
request injection channel -- Producer -> C++ submit/close/EOF/error always
go through the step-1-2 command queue and never touch req_notify.fifo.

Protocol v1 (frozen rules, written into contract ②/④ and
online_contracts/state_delta_v1.md; phase 4 §7.1):
  - Wire: <run_dir>/bridge/ with req_notify.fifo / resp_notify.fifo
    (mkfifo). FIFO open order: the bridge directory and both FIFOs are
    created (ensure_bridge_dir) BEFORE either side starts.
  - Request:  C++ writes request_<seq>.json (atomic: tmp + rename), then
    writes ONE byte to req_notify.fifo. C++ holds the req_notify write end
    open for the bridge's lifetime, so the Python reader blocks in read(1)
    between notifications and only sees EOF when the run ends.
  - Response: Python writes response_<seq>.json (atomic), then writes ONE
    byte to resp_notify.fifo. Defect-B fix (2026-08-16): BOTH resp ends are
    LONG-LIVED -- Python opens its write end once at serve_forever start
    (pairs with C++'s read end from open_notify) and holds it until exit;
    C++ polls its lifetime read end. The old per-exchange open/write/close
    and open/poll/read/close handshake had an unfixable kernel race family
    (POLLHUP+read()==0 false "crash" when a fresh read fd's poll_wait
    caught the writer's close transition with an empty buffer; mirror-race
    EPIPE on Python's write) -- with no mid-run fd transitions, read()==0
    can only mean the peer process died, and byte count stays exactly
    1:1 with deliveries (backpressure: one in-flight epoch at a time).
  - seq == delivery_sequence (monotonic epoch number), so request /
    response / commit_ack file names for one epoch all share the same seq.
  - Request fields:  {schema_version, delivery_sequence, delivery_epoch,
    tick, deferred_from_tick, reasons[], arrivals[], completed_groups[],
    completed_nodes[], retry_items[], affected_ranks[], snapshot_handle,
    snapshot:{}, ledger_summary}. reasons[] holds one reason name per event
    in epoch order (ARRIVAL/PREFILL_DRAIN/DECODE_COMPLETION/
    REQUEST_COMPLETE); arrivals[] carries the ARRIVAL envelope facts
    (incl. ingress_seq/queue_index, v1); completed_groups[] carries the
    watch-fire facts (request_id/stage/generation/node_count);
    completed_nodes[] the per-node terminal facts (v1);
    retry_items[] is always empty in v1; affected_ranks[] the epoch's
    affected rank set (v1); snapshot_handle the placeholder (v1, expiry
    rule frozen in the contract); snapshot is the v0 reserved field kept
    empty for backward comprehension.
  - Response fields: {schema_version, batch_id, source_delivery_sequence,
    nodes[], parent_edges[], watches[], assignments[], kv_actions[],
    future_alarms[], error?}. The inner node/edge/watch/assignment/kv-action
    schemas are carried opaquely by C++ (nlohmann::json arrays) and freeze
    with step 1-8's graph_batch_builder; the step-1-11 committer parses
    them. future_alarms[] (step 1-8) carries {arrival_world_ns, envelope{
    request_id, session_id, turn_index, prefill_length, decode_length,
    inter_request_interval_ns}} -- the committer schedules these through
    RequestIngress::schedule_future_arrival inside the commit.
  - Error semantics: a Python exception (or protocol failure) => response
    WITH the error field set, notify, then Python exits non-zero; C++
    reads the error and aborts (fail-closed).
  - Python crash detection (defect-B fix semantics): EOF (zero bytes) on
    the LIFETIME resp_notify read end, or EPIPE on the req_notify write,
    means the Python process is gone -> C++ aborts with a clear message.
    Because both resp ends are long-lived, "no writer yet" states no longer
    exist mid-run and cannot be mistaken for death. A poll timeout
    (timeout_ms > 0) aborts identically.
  - Duplicate seq idempotency: Python tracks processed request seqs and
    ack seqs separately and ignores duplicates (files are atomic, so a
    duplicate can only come from a retry).
  - Backpressure: one in-flight request at most -- C++ never sends
    request_<seq+1> before response_<seq> is consumed; a commit ack can be
    sent right after the response (Python is back in read(1) by then).
  - Cleanup: ensure_bridge_dir removes stale request_*.json,
    response_*.json and commit_ack_*.json files (and *.tmp) from a previous
    run before both sides start.
*******************************************************************************/

#ifndef EXECUTION_DRIVEN_DECISIONBRIDGE_HH
#define EXECUTION_DRIVEN_DECISIONBRIDGE_HH

#include <cstdint>
#include <string>

#include <json/json.hpp>

#include "astra-sim/workload/execution_driven/DecisionMailbox.hh"
#include "astra-sim/workload/execution_driven/ParsedGraphBatch.hh"

namespace AstraSim {
namespace ExecutionDriven {

/// Bridge protocol schema version (frozen, v1 -- phase 4 §7.1;
/// online_contracts/state_delta_v1.md is the authority).
inline constexpr int kDecisionBridgeSchemaVersion = 1;

/// GraphBatch: the C++<-Python reply of one delivery epoch.
/// C1 (2026-08-29): the batch is TYPED. The pre-C1 struct carried the six
/// nlohmann::json arrays across the bridge->committer boundary and every
/// consumer (validate, liveness preflight, commit assembly, anchor
/// registration) re-extracted the fields from the DOM -- four full walks per
/// batch. deliver_and_receive now parses the response ONCE via
/// parse_graph_batch (ParsedGraphBatch.hh; structural fail-closed rules
/// T/N/E/W/A/S/O) and the DOM dies before the bridge call returns. The
/// historical name survives as an alias so downstream code and fixtures
/// keep compiling; ParsedGraphBatch is the canonical type.
///   - nodes/parent_edges/watches/assignments/kv_actions/future_alarms:
///     typed vectors in ARRAY ORDER (the emission order; never re-sorted).
///   - touched_ranks (phase 5): the Python-computed rank set of the batch's
///     nodes (sorted unique); the committer validates it against its own
///     computation (has_touched_ranks distinguishes an absent field --
///     tolerated for pre-phase-5 fixtures -- from an empty one, the legal
///     value of a zero-node batch).
///   - error: non-empty means the decision failed; the bridge aborts on it
///     BEFORE the structural parse (the frozen error skeleton carries
///     all-empty arrays, so the message must be the decision error).
using GraphBatch = ParsedGraphBatch;

/// Serialize one StateDelta into the v1 request JSON (protocol contract,
/// online_contracts/state_delta_v1.md; exposed for fixtures and the
/// step-1-8 wiring).
nlohmann::json build_request_json(const StateDelta& delta);

/// Decision channel interface (方案 §4 步骤 1-7 操作 2).
class DecisionBridge {
  public:
    virtual ~DecisionBridge() = default;

    /// Deliver one StateDelta to Python and block until the GraphBatch
    /// reply arrives. Exactly one delivery per tick (backpressure).
    virtual GraphBatch deliver_and_receive(const StateDelta& delta) = 0;

    /// Notify Python that the batch with batch_id (epoch seq) was
    /// committed. Python's provisional ledger finalizes on this ack
    /// (step-1-6 pseudocode; the ledger itself lands with step 1-8/1-9).
    virtual void send_commit_ack(uint64_t batch_id, uint64_t delivery_seq,
                                 bool success) = 0;
};

/// File-implementation of the v0 protocol (blocking FIFOs, atomic JSON
/// files, fail-closed error/crash/timeout semantics). timeout_ms == 0
/// waits forever (the phase-1 default).
/// C1 (2026-08-29): num_ranks feeds parse_graph_batch's rank-domain checks
/// (node/edge/watch-member/touched-rank ranges). -1 (the default) disables
/// them for legacy fixtures; the official online main passes its NPU count.
class FileDecisionBridge : public DecisionBridge {
  public:
    explicit FileDecisionBridge(std::string bridge_dir, int timeout_ms = 0,
                                int num_ranks = -1);
    ~FileDecisionBridge() override;

    FileDecisionBridge(const FileDecisionBridge&) = delete;
    FileDecisionBridge& operator=(const FileDecisionBridge&) = delete;

    GraphBatch deliver_and_receive(const StateDelta& delta) override;
    void send_commit_ack(uint64_t batch_id, uint64_t delivery_seq,
                         bool success) override;

    /// Create <bridge_dir>, both FIFOs and clean stale files. Must be
    /// called BEFORE either side starts (FIFO open order rule).
    static void ensure_bridge_dir(const std::string& bridge_dir);

    /// Open the req_notify write end and hold it for the run lifetime
    /// (step 1-10: zero-delivery runs must still give Python a clean EOF).
    /// O_WRONLY|O_NONBLOCK on a FIFO with no reader fails with ENXIO, so it
    /// retries until the Python reader's blocking open lands (bounded by
    /// timeout_ms_ when > 0). Must be called right after ensure_bridge_dir,
    /// BEFORE the main loop (the Python side starts from a separate process
    /// once the FIFOs exist).
    ///
    /// Defect-B fix (2026-08-16, face主动测试错误分析.md 缺陷 B): this now
    /// ALSO opens the resp_notify READ end once and holds it for the bridge
    /// lifetime (mirroring the req channel's long-lived design). The old
    /// per-exchange open/poll/read/close handshake had an unfixable kernel
    /// race family (a fresh read fd whose poll_wait catches the writer's
    /// 1->0 close transition with an empty buffer wakes with POLLHUP and
    /// read()==0 -- the false "Python side crashed"; the mirror variant
    /// EPIPEs Python's write after a reader wake/close race). With both
    /// ends long-lived, read()==0 on this fd can ONLY mean the Python
    /// process died (its lifetime write end closed) -- EOF semantics become
    /// true. Python holds the write end from serve_forever start to exit.
    void open_notify();

    /// Phase-6 (方案 §9.1) per-run channel stats. Cumulative across the run;
    /// zero for bridges that never deliver. Measured from the C++ side:
    ///   - roundtrip_count / roundtrip_ns: deliver_and_receive calls and the
    ///     cumulative wall ns of the full blocking round trip (write request
    ///     + notify + wait for Python + read response);
    ///   - channel_bytes: JSON payload bytes written AND read through the
    ///     channel (request_<seq>.json + response_<seq>.json +
    ///     commit_ack_<seq>.json; the C++ side counts its own dump size,
    ///     which equals the on-disk file bytes);
    ///   - forced_flush_count: one per atomic JSON file write
    ///     (write_file_atomic's out.flush(); 方案 §15: 正式日志无逐节点强制
    ///     flush -- 该计数必须为 O(deliveries) 而非 O(nodes)).
    struct Stats {
        uint64_t roundtrip_count = 0;
        uint64_t roundtrip_ns = 0;
        uint64_t channel_bytes = 0;
        uint64_t forced_flush_count = 0;
    };

    const Stats& stats() const {
        return stats_;
    }

    /// One-line report ("[online] phase-6 stats counters:" continuation).
    std::string stats_report() const;

  private:
    std::string request_path(uint64_t seq) const;
    std::string response_path(uint64_t seq) const;
    std::string ack_path(uint64_t seq) const;

    void notify_python();  // one byte on req_notify.fifo (EPIPE => abort)
    void write_file_atomic(const std::string& path,
                           const nlohmann::json& payload) const;
    /// Defect-B fix: poll + read exactly ONE byte off the long-lived
    /// resp_notify read end. read()==0 => Python's lifetime write end
    /// closed => genuine peer death. EAGAIN (spurious wake before the
    /// byte) re-enters the poll. Returns false only via bridge_fatal.
    void wait_response_byte(uint64_t seq);

    std::string bridge_dir_;
    int timeout_ms_;
    int num_ranks_ = -1;  // C1: rank-domain checks in parse_graph_batch
    int req_notify_fd_ = -1;  // write end, held open for the run lifetime
    // Defect-B fix (2026-08-16): resp_notify read end, held open for the
    // run lifetime (see open_notify). -1 until open_notify/lazy open.
    int resp_notify_fd_ = -1;
    // Mutable: write_file_atomic() is const (the phase-1 contract) but the
    // phase-6 channel accounting is a per-run observation, not bridge state.
    mutable Stats stats_;
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_DECISIONBRIDGE_HH
