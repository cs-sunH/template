/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

DecisionBridge -- execution-driven mechanism layer (wscllm phase 1).
Interface + shared-memory implementation (方案 §4 步骤 1-7; joint v11 §8
保底: ShmFrameChannel carrier, no loose frame files).

The C++ <-> Python decision channel of the Execution-Driven loop. The
tick-end gate (ed_driver_tick_end) drains the DecisionMailbox into one
StateDelta and calls deliver_and_receive(delta); the returned GraphBatch is
the scheduler's reply. This bridge is a DECISION channel, NOT a runtime
request injection channel -- Producer -> C++ submit/close/EOF/error always
go through the step-1-2 command queue and never touch req_notify.fifo.

Protocol v1 (frozen rules, written into contract ②/④; phase 4 §7.1;
v11 §8 保底载体替换 -- shm frames instead of loose JSON files, byte-level
semantics unchanged):
  - Wire: <run_dir>/bridge/ with req_notify.fifo / resp_notify.fifo
    (mkfifo), the audit journal files (request_journal.jsonl + the gz
    sidecars, written by Python) and bridge_shm.json (the C++-written
    pointer to this run's shm segments). FIFO open order: the bridge
    directory, both FIFOs and the shm segments are created
    (ensure_bridge_dir) BEFORE either side starts.
  - Carrier: three POSIX-shm single-slot frame channels (ShmFrameChannel;
    names inside bridge_shm.json): request (C++ -> Python), commit_ack
    (C++ -> Python) and response (Python -> C++). A frame is
    [u32 magic][u64 seq][u32 len][payload][u32 crc32] (little-endian,
    IEEE crc32 over header||payload); payload bytes are EXACTLY the old
    protocol's file bytes (request_<seq>.json / response_<seq>.json /
    commit_ack_<seq>.json content; the work_ledger splice still happens in
    C++ before the frame is published). Oversize/corrupt/crc mismatch are
    protocol violations -> fail closed (same channel as every other
    violation). The segments carry a startup generation; Python validates
    it against bridge_shm.json, so a stale segment from a crashed previous
    run can never be mistaken for this run's (cross-run pollution guard).
    Doorbell synchronization: the FIFO byte written AFTER the full frame
    publish is the reader's only wake-up -- no atomics are needed, and the
    1:1 backpressure guarantees a published frame is never overwritten
    before the peer consumed it.
  - Request:  C++ publishes the request frame, then writes ONE byte to
    req_notify.fifo. C++ holds the req_notify write end open for the
    bridge's lifetime, so the Python reader blocks in read(1) between
    notifications and only sees EOF when the run ends. Python journals the
    request bytes (audit stream unchanged; the loose request_<seq>.json
    files no longer exist).
  - Response: Python publishes the response frame (bytes identical to the
    old response_<seq>.json), then writes ONE byte to resp_notify.fifo.
    Defect-B fix (2026-08-16): BOTH resp ends are LONG-LIVED -- Python
    opens its write end once at serve_forever start (pairs with C++'s read
    end from open_notify) and holds it until exit; C++ polls its lifetime
    read end. With no mid-run fd transitions, read()==0 can only mean the
    peer process died, and byte count stays exactly 1:1 with deliveries
    (backpressure: one in-flight epoch at a time).
  - seq == delivery_sequence (monotonic epoch number), so the request /
    response / commit_ack frames of one epoch all share the same seq.
    Request and ack sequences are tracked separately on Python (a commit
    ack for epoch N can be pending while request N+1 is in flight; the
    serve loop processes acks before requests, exactly like the old
    file-scan wake).
  - Request fields:  {schema_version, delivery_sequence, delivery_epoch,
    tick, deferred_from_tick, reasons[], arrivals[], completed_groups[],
    completed_nodes[], affected_ranks[], snapshot_handle,
    snapshot:{}, ledger_summary}. reasons[] holds one reason name per event
    in epoch order (ARRIVAL/PREFILL_DRAIN/DECODE_COMPLETION/
    REQUEST_COMPLETE); arrivals[] carries the ARRIVAL envelope facts
    (incl. ingress_seq/queue_index, v1); completed_groups[] carries the
    watch-fire facts (request_id/stage/generation/node_count);
    completed_nodes[] the per-node terminal facts (v1);
    affected_ranks[] the epoch's affected rank set (v1); snapshot_handle
    the placeholder (v1, expiry rule frozen in the contract); snapshot is
    the v0 reserved field kept empty for backward comprehension.
    C6 (WP2, --link-telemetry, default OFF): when the online main installs
    the LinkTelemetryProvider, requests additionally carry the top-level
    link_telemetry[] array (additive, sibling of ledger_summary; see
    LinkTelemetryProvider for the element schema and the F5/F6 semantics).
    Flag off = field absent; the Python side only pins schema_version and
    tolerates the additive field either way.
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
    frame WITH the error field set, notify, then Python exits non-zero;
    C++ reads the error and aborts (fail-closed).
  - Python crash detection (defect-B fix semantics): EOF (zero bytes) on
    the LIFETIME resp_notify read end, or EPIPE on the req_notify write,
    means the Python process is gone -> C++ aborts with a clear message.
    Because both resp ends are long-lived, "no writer yet" states no longer
    exist mid-run and cannot be mistaken for death. A poll timeout
    (timeout_ms > 0) aborts identically.
  - Duplicate seq idempotency: Python tracks processed request seqs and
    ack seqs separately and ignores duplicates (a duplicate doorbell byte
    finds the slot still holding the already-consumed seq and skips it --
    the same ignore the loose-file contains() filter performed).
  - Backpressure: one in-flight request at most -- C++ never publishes
    request <seq+1> before the response <seq> frame is consumed; a commit
    ack can be sent right after the response (Python is back in read(1) by
    then).
  - Cleanup: ensure_bridge_dir removes stale request_*.json,
    response_*.json and commit_ack_*.json files (and *.tmp) from a previous
    run and unlinks stale shm segments carrying this bridge dir's stem
    prefix before creating the new generation; the bridge destructor
    unlinks this run's segments (mappings held by a still-exiting Python
    stay valid). A crashed C++ leaves the segments behind -- abort() runs
    no destructors -- which doubles as the post-mortem evidence the old
    fail-closed path's surviving response file provided.
*******************************************************************************/

#ifndef EXECUTION_DRIVEN_DECISIONBRIDGE_HH
#define EXECUTION_DRIVEN_DECISIONBRIDGE_HH

#include <cstdint>
#include <functional>
#include <string>

#include <json/json.hpp>

#include "astra-sim/workload/execution_driven/DecisionMailbox.hh"
#include "astra-sim/workload/execution_driven/ParsedGraphBatch.hh"
#include "astra-sim/workload/execution_driven/ShmFrameChannel.hh"

namespace AstraSim {
namespace ExecutionDriven {

/// Bridge protocol schema version (frozen, v1 -- phase 4 §7.1; this
/// header and the Python validator are the authority).
inline constexpr int kDecisionBridgeSchemaVersion = kGraphBatchSchemaVersion;

/// C6 (WP2 link telemetry, joint 遥测改造, --link-telemetry): optional
/// per-delivery top-level link_telemetry[] payload provider, installed by
/// the online main when the flag is on. Each element carries
/// {link_id, served_bytes, active_ns, window_start_ns, window_end_ns}:
///   - window = one decision epoch's physical duration (F5: adaptive, no
///     new constants; window bounds are the previous and current delivery
///     ticks);
///   - served_bytes/active_ns = differentials of the fluid link
///     observer's cumulative totals (const link_observer_totals());
///   - coverage (F6): NoC leg only -- collectives included (every
///     collective routes sim_send -> start_flow), pool ports excluded
///     (Workload issue_remote_mem -> AnalyticalRemoteMemory; registry
///     model owns that contention).
/// The provider runs inside deliver_and_receive on the simulation thread,
/// exactly once per delivery epoch (the 1:1 backpressure guarantees no
/// double sampling). Unset provider = the field is omitted entirely, so
/// flag-off runs serialize byte-identical requests (fail-closed default;
/// StateDelta and the frozen v1 schema fields are untouched -- the array
/// rides as an additive top-level sibling of ledger_summary and never
/// occupies a snapshot reserved slot).
using LinkTelemetryProvider =
    std::function<nlohmann::json(const StateDelta&)>;

/// W1 (joint work ledger, 2026-09-29): per-delivery top-level `work_ledger`
/// payload provider, installed by the online main. Same additive-sibling
/// discipline as LinkTelemetryProvider ( rides beside ledger_summary, never
/// occupies a snapshot reserved slot; absent = legacy peer, the Python side
/// records the absence rather than fabricating zeros). The payload shape is
/// frozen by the W1 schema contract (BRIDGE_PROTOCOL_VERSION
/// "bridge-v1-20260929" / WORK_LEDGER_SCHEMA_VERSION "workledger-v1-20260929"
/// on the Python side, joint/schema_registry.py):
///
///   { "schema_version": 1,
///     "bridge_protocol": "bridge-v1-20260929",
///     "work_ledger_schema": "workledger-v1-20260929",
///     "snapshot_kind": "full" | "delta",
///     "delivery_sequence": <epoch>, "tick": <tick>,
///     "event_watermark": <monotonic events assigned so far>,
///     "state": {...}            -- full mode: the whole ledger state
///     "events": [...] }         -- delta mode: diff-derived lifecycle events
///
/// State-derived ledger (inventory §4.3 point 4): the provider emits the
/// FULL state on the first delivery and, on every later one, the
/// diff_work_ledger_states() event list derived from the previous frame --
/// commit/issue/progress/completion/ack/release with monotonic local_seq.
/// The provider runs inside deliver_and_receive on the simulation thread,
/// exactly once per delivery epoch (1:1 backpressure).
// v6 compact round: the provider returns the encoded workledger-compact-v1
// frame (deterministic bytes + the header facts the timing surface reads).
struct WorkLedgerWirePayload {
    std::string bytes;
};
using WorkLedgerProvider =
    std::function<WorkLedgerWirePayload(const StateDelta&)>;

/// Plan D (request-independent periodic thermal epoch, 2026-10-20): optional
/// per-delivery top-level `thermal_activity` provider, installed by the
/// online main when SH_THERMAL_EPOCH_NS > 0. Same additive-sibling
/// discipline as LinkTelemetryProvider/WorkLedgerProvider (rides beside
/// ledger_summary, never occupies a snapshot reserved slot; absent = the
/// thermal feature is off, so the serialized request bytes are identical to
/// the pre-D wire format). The frozen payload shape (C++ <-> Python):
///
///   { "schema_version": 1,
///     "epoch_ns": <int>,         -- the simulation instant this event takes
///                                   effect
///     "window_start_ns": <int>,  -- the previous thermal-epoch instant (0 on
///                                   the first epoch)
///     "window_end_ns": <int>,    -- == epoch_ns
///     "coverage": "measured",
///     "provenance": "coarse_runtime_counters_delta",
///     "sources": [ { "source_id": "die<N>.comp", "action_kind": "pe",
///                    "count": <number> }, ... ] }  // may be empty
///
/// The provider runs inside deliver_and_receive on the simulation thread,
/// exactly once per delivery epoch (1:1 backpressure guarantees no double
/// sampling), driving the differential of the coarse die runtime's cumulative
/// served counters. sources may legitimately be empty (nothing served in the
/// window, or no coarse die runtime on this rank) -- the C++ side never
/// fabricates activity. The `die<N>` identity is the THERMAL die number, which
/// only the cooling side's mapping knows; this build has no such map in C++,
/// so it carries the RANK number as an explicit placeholder (see
/// main_online.cc compute_thermal_activity) -- Python rewrites it through the
/// map and never trusts the placeholder as a thermal-die fact.
using ThermalActivityProvider =
    std::function<nlohmann::json(const StateDelta&)>;

/// Plan D response consumer: receives the parsed response's `thermal_cap`
/// member verbatim when the request carried `thermal_activity` and the
/// Python side answered with a cap. Installed by the online main (which owns
/// the CoarseDieRuntime objects); the bridge NEVER depends on
/// CoarseDieRuntime itself (layer decoupling). The frozen payload shape:
///
///   { "schema_version": 1,
///     "resource_epoch": <int>,
///     "effective_from_ns": <int>,
///     "events": [ { "resource_index": <int 0..12>,
///                   "absolute_physical_rate": <number >= 0> } ] }
///
/// resource_index is the coarse-die resource-table index (COMPUTE_GATE=0;
/// see extern/die_model/coarse/src/Hardware.cc and
/// online/coarse_runtime_client.py RESOURCE_IDS). The consumer is invoked
/// inside deliver_and_receive after the structural parse, on the simulation
/// thread. A cap that is absent (the Python peer has no cap yet, or the
/// decision was a no-op) is simply not delivered -- the bridge records
/// nothing and continues (never fails closed on absence).
using ThermalCapConsumer = std::function<void(const nlohmann::json&)>;

/// C4 protocol-2 lifecycle publication hook. The caller assigns the one
/// run-global logical sequence at the instant this callback is made. The
/// NodeView is the authoritative committed online record; node_id is its
/// NodeStore ID (not the zero-based GraphBatch JSON ID).
using LedgerNodeTransitionPublisher = std::function<void(
    const std::string& kind, int rank, uint64_t node_id,
    const OnlineNode& node, const std::vector<uint64_t>& parent_store_ids)>;

/// W1: derive the incremental ledger delta between two work-ledger state
/// frames (pure JSON -> JSON; no simulator state is touched).
/// `next_local_seq` is the producer's monotonic counter: every emitted
/// event consumes one value and the counter advances past the last used
/// value, so consecutive calls never reuse a (producer, local_seq)
/// identity.
///
/// Diff rules (state-derived six-kind lifecycle):
///   - key only in cur        -> commit (+ issue/completion replay of the
///                               current flags for a mid-life first sight);
///   - issued false -> true   -> issue;
///   - finished false -> true -> completion;
///   - remaining_ops/bytes/latency_ns decreased -> progress (delta_served =
///     the decrease, plus the corresponding current remaining_* field from
///     the same tick-end frame; source is inferred on the Python side);
///   - key only in prev       -> completion (if not already finished) then
///     ack then release (NodeStore GC only collects finished nodes whose
///     children all finished, so disappearance means the obligations
///     settled).
///
/// Compact node lists: node-class lifecycle changes (the dominant volume --
/// a busy decision epoch commits thousands of graph nodes) are emitted as
/// four id/fact lists instead of per-event envelopes:
///   compact.node_commits     -- full node entries (same shape as
///                               state.nodes[].nodes[] elements);
///   compact.node_issues      -- work ids whose issued flag flipped true;
///   compact.node_completions -- work ids whose finished flag flipped true;
///   compact.node_vanished    -- work ids absent from the current frame
///                               (the Python side emits completion-if-needed
///                               then release; no separate ack -- release
///                               subsumes it for the state-derived ledger);
///   compact.seq_base         -- the first local_seq the reconstructed
///                               node events take (the COMPACT entries
///                               consume seqs from seq_base FIRST, in list
///                               order commits -> issues -> completions ->
///                               vanished at 2 seqs each; the `events`
///                               array carries the C++-assigned HIGHER
///                               seqs after them -- mirroring the Python
///                               application order in joint/work_ledger.py
///                               and the seq_base arithmetic in
///                               DecisionBridge.cc; 2026-10-01 审查订正：
///                               旧注释把两序写反，按其"修复"公式即破坏
///                               (producer, local_seq) 配对).
///   Invariant (2026-10-01 R2 注记): the compact-first ordering holds only
///   while node entries carry NO remaining_ops/remaining_bytes/
///   remaining_latency_ns fields -- a node entry with any of them emits a
///   progress event that consumes a seq BETWEEN the compact and events
///   segments and silently breaks the pairing. Keep diff_one_section's
///   node serializer free of remaining_* keys (or extend the seq layout
///   deliberately).
/// The Python ledger reconstructs the same six-kind LedgerEvents from these
/// lists (deterministic order), so full-vs-incremental equivalence and the
/// idempotency gate are unchanged.
///
/// Section order (nodes, hbm, pool, noc flows) and per-section key order
/// are deterministic; the caller must build both frames with the same
/// collector so the maps are comparable.
struct WorkLedgerDiff {
    nlohmann::json events;   // leg works + progress (full event form)
    nlohmann::json compact;  // node-class lists (see above); object when
                             // non-empty, null when nothing changed
};

WorkLedgerDiff diff_work_ledger_states(const nlohmann::json& prev_state,
                                       const nlohmann::json& cur_state,
                                       uint64_t delivery_sequence,
                                       uint64_t tick,
                                       uint64_t& next_local_seq);

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

/// Serialize one StateDelta into the v1 request JSON (protocol contract;
/// exposed for fixtures and the step-1-8 wiring).
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

/// Shared-memory implementation of the v1 protocol (blocking FIFO
/// doorbells, single-slot shm frame channels, fail-closed error/crash/
/// timeout semantics; joint v11 §8 保底 -- direct carrier replacement of
/// the former file-based transport, no runtime switch). timeout_ms == 0
/// waits forever (the phase-1 default).
/// C1 (2026-08-29): num_ranks feeds parse_graph_batch's rank-domain checks
/// (node/edge/watch-member/touched-rank ranges). -1 (the default) disables
/// them for legacy fixtures; the official online main passes its NPU count.
class ShmDecisionBridge : public DecisionBridge {
  public:
    explicit ShmDecisionBridge(std::string bridge_dir, int timeout_ms = 0,
                               int num_ranks = -1);
    ~ShmDecisionBridge() override;

    ShmDecisionBridge(const ShmDecisionBridge&) = delete;
    ShmDecisionBridge& operator=(const ShmDecisionBridge&) = delete;

    GraphBatch deliver_and_receive(const StateDelta& delta) override;
    void send_commit_ack(uint64_t batch_id, uint64_t delivery_seq,
                         bool success) override;

    /// Create <bridge_dir>, both FIFOs, the shm frame segments (fresh
    /// generation; stale segments of this bridge dir are unlinked) and
    /// bridge_shm.json, and clean stale loose frame files. Must be called
    /// BEFORE either side starts (FIFO open order rule).
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
    /// lifetime (mirrors the req channel's long-lived design). The old
    /// per-exchange open/poll/read/close handshake had an unfixable kernel
    /// race family (a fresh read fd whose poll_wait catches the writer's
    /// 1->0 close transition with an empty buffer wakes with POLLHUP and
    /// read()==0 -- the false "Python side crashed"; the mirror variant
    /// EPIPEs Python's write after a reader wake/close race). With both
    /// ends long-lived, read()==0 on this fd can ONLY mean the Python
    /// process died (its lifetime write end closed) -- EOF semantics become
    /// true. Python holds the write end from serve_forever start to exit.
    void open_notify();

    /// C6 (WP2, --link-telemetry): install the link telemetry provider.
    /// Once installed, every request frame gains the top-level
    /// link_telemetry[] array (the provider's return value; see
    /// LinkTelemetryProvider). Never installed = the field is absent.
    /// Legal only before the first delivery (the main installs it at
    /// startup wiring; a mid-run swap would change the request schema
    /// between epochs of one run).
    void set_link_telemetry_provider(LinkTelemetryProvider provider);

    /// W1 (joint work ledger): install the work-ledger provider. Once
    /// installed, every request frame gains the top-level `work_ledger`
    /// member (see WorkLedgerProvider). Never installed = the field is
    /// absent (legacy peer). Same startup-only contract as the telemetry
    /// provider. The two channels are independent: the ledger never gates on
    /// --link-telemetry and carries its own schema versions.
    void set_work_ledger_provider(WorkLedgerProvider provider);

    /// Plan D (periodic thermal epoch): install the top-level `thermal_activity`
    /// provider. Once installed, every request frame gains the additive
    /// thermal_activity sibling (see ThermalActivityProvider). Never installed
    /// (SH_THERMAL_EPOCH_NS == 0, the default) = the field is absent and the
    /// request bytes are byte-identical to the pre-D wire format. Same
    /// startup-only contract as the telemetry/ledger providers.
    void set_thermal_activity_provider(ThermalActivityProvider provider);

    /// Plan D: install the response `thermal_cap` consumer (see
    /// ThermalCapConsumer). Installed alongside the provider by the online
    /// main; the bridge itself stays free of CoarseDieRuntime. Same
    /// startup-only contract.
    void set_thermal_cap_consumer(ThermalCapConsumer consumer);

    /// Phase-6 (方案 §9.1) per-run channel stats. Cumulative across the run;
    /// zero for bridges that never deliver. Measured from the C++ side:
    ///   - roundtrip_count / roundtrip_ns: deliver_and_receive calls and the
    ///     cumulative wall ns of the full blocking round trip (publish
    ///     request frame + notify + wait for Python + read response frame);
    ///   - channel_bytes: payload bytes that crossed the channels
    ///     (request + response + commit_ack frame payloads -- exactly the
    ///     byte counts the old file transport measured as written body
    ///     bytes / on-disk file size; B1 2026-08-23 accounting preserved);
    ///   - forced_flush_count: one per published frame on this side
    ///     (request + commit_ack; 方案 §15: 正式日志无逐节点强制 flush --
    ///     该计数必须为 O(deliveries) 而非 O(nodes)). One shm publish counts
    ///     as one flush (the accounting surface of the old atomic write).
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

    // -- shm transport facts (exposed for the in-process loopback tests;
    //    production Python reads the same values from bridge_shm.json) --
    /// The startup generation of this run's segments.
    uint64_t shm_generation() const {
        return shm_generation_;
    }
    /// Payload capacity of one frame slot.
    uint64_t shm_slot_capacity() const {
        return shm_slot_capacity_;
    }
    /// Segment names without the leading '/' (as written to
    /// bridge_shm.json): request / commit-ack / response channels.
    const std::string& shm_request_segment() const {
        return shm_request_segment_;
    }
    const std::string& shm_commit_ack_segment() const {
        return shm_commit_ack_segment_;
    }
    const std::string& shm_response_segment() const {
        return shm_response_segment_;
    }

  private:
    /// Create (fresh generation, stale segments unlinked) -- or leave
    /// alone when already valid -- this bridge dir's segments and the
    /// bridge_shm.json pointer. Static: holds no mappings on return
    /// (segment unlinking belongs to the instance destructor only).
    static void mint_bridge_shm_info(const std::string& bridge_dir);

    /// Ensure the segments are minted (see mint_bridge_shm_info) and map
    /// them into this bridge. Idempotent per instance; fail-closed on any
    /// inconsistency.
    void ensure_bridge_shm();

    /// Publish the request frame; publish failures (oversize) fail closed.
    void publish_request_frame(uint64_t seq, const std::string& body);

    void notify_python();  // one byte on req_notify.fifo (EPIPE => abort)
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
    // Mutable: publish paths are const-clean (the phase-1 contract) but the
    // phase-6 channel accounting is a per-run observation, not bridge state.
    mutable Stats stats_;
    // C6 (WP2, --link-telemetry): empty = telemetry off (field omitted).
    LinkTelemetryProvider link_telemetry_provider_;
    // W1 (joint work ledger): empty = ledger off (field omitted).
    WorkLedgerProvider work_ledger_provider_;
    // Plan D (periodic thermal epoch): empty = thermal off (field omitted).
    ThermalActivityProvider thermal_activity_provider_;
    // Plan D: empty = no cap consumer (a delivered cap is then ignored).
    ThermalCapConsumer thermal_cap_consumer_;
    // v11 §8 shm carrier: three single-slot frame channels + the segment
    // identity (also published to Python via bridge_shm.json).
    uint64_t shm_generation_ = 0;
    uint64_t shm_slot_capacity_ = 0;
    std::string shm_request_segment_;
    std::string shm_commit_ack_segment_;
    std::string shm_response_segment_;
    ShmFrameChannel req_channel_;   // C++ -> Python (request frames)
    ShmAckRing ack_channel_;        // C++ -> Python (commit-ack ring; the
                                    // ack stream is not exchange-gated)
    ShmFrameChannel resp_channel_;  // Python -> C++ (response frames)
    bool shm_prepared_ = false;     // segments adopted/created + mapped
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_DECISIONBRIDGE_HH
