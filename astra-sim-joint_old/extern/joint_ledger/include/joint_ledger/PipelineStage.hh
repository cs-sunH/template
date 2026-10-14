// PipelineStage.hh -- native port of the pipeline ledger staging/commit core
// (pipeline_ledger.py PipelineLedger/PipelineLedgerV4: _gate_delivery,
// _stage_delivery, commit_prepared and the query surface).
//
// prepare() validates global logical order, stages every transition through
// COW overlays (rows, counts, indexes, terminal-provenance treaps) and
// drives the coordinated WorkLedger transaction for node events.  commit()
// publishes the fully validated update; a prepare failure leaves every
// overlay discarded (the caller destroys the prepared slot).
//
// FILE OWNERSHIP: the public API below is the frozen cross-module contract;
// the private implementation section is owned by the stage-core port unit
// and may be restructured freely (row containers, treap nodes, overlays).
#ifndef JOINT_LEDGER_PIPELINE_STAGE_HH
#define JOINT_LEDGER_PIPELINE_STAGE_HH

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "LedgerTypes.hh"
#include "PipelineDescriptor.hh"
#include "PipelineRows.hh"
#include "WorkLedgerCore.hh"

namespace jointledger {

struct PipelineReceipt {
    std::string snapshot_kind;
    std::uint64_t delivery_sequence = 0;
    std::uint64_t logical_event_count = 0;
    std::uint64_t base_watermark = 0;
    std::uint64_t event_watermark = 0;
    std::uint64_t base_state_version = 0;
    std::uint64_t state_version = 0;
    bool replayed = false;
};

class PipelineLedgerCore {
 public:
    PipelineLedgerCore();
    ~PipelineLedgerCore();                      // defined with Impl complete

    // ---- delivery path ----
    // Mirrors prepare_envelope (gate + global-sequence merge + _stage_
    // delivery).  Returns a prepared slot id for commit_prepared(); the
    // update is replayed iff parsed is an idempotent replay of the last
    // accepted delivery (checked with the raw segment bytes of THIS
    // delivery; a same-sequence delivery with different raw bytes fails
    // closed -- the deterministic producer never reformats identical
    // content).
    int prepare(const ParsedEnvelope& parsed, WorkLedgerCore::Transaction& tx,
                const std::string& segment_bytes);
    PipelineReceipt commit_prepared(int prepared_slot);
    void discard_prepared(int prepared_slot);

    // ---- query surface ----
    // pricing_snapshot() shape: {"epoch", "delivery_sequence",
    // "event_watermark", "state_version", "tick", "active_resources": [...]}
    Json pricing_json() const;
    // remaining_view()/snapshot_state() shape for offline harness/digests.
    Json snapshot_state_json() const;

    bool initialized() const { return initialized_; }
    const std::string& epoch() const { return epoch_; }
    std::uint64_t delivery_sequence() const { return delivery_sequence_; }
    std::uint64_t watermark() const { return watermark_; }
    std::uint64_t state_version() const { return state_version_; }
    std::int64_t last_tick() const { return last_tick_; }
    const std::string& last_replay_token() const { return last_replay_token_; }
    std::uint64_t replay_count() const { return replay_count_; }
    // Raw bytes of the last accepted delivery's work_ledger segment (the
    // idempotent-replay gate cache).
    const std::string& last_content_raw() const { return last_content_raw_; }

    // Retained-registry view consumed by parse_envelope_v4.
    const std::unordered_map<std::string, DescriptorPtr>& descriptors() const;
    const std::unordered_map<std::string, SelectorIndex>& selector_indexes()
        const;

 private:
    struct Impl;                       // stage-core-owned state
    std::unique_ptr<Impl> impl_;

    // Small scalar gate fields kept inline for cheap accessor dispatch.
    bool initialized_ = false;
    std::string epoch_;
    std::int64_t delivery_sequence_ = -1;
    std::uint64_t watermark_ = 0;
    std::uint64_t state_version_ = 0;
    std::int64_t last_tick_ = -1;
    std::string last_replay_token_;
    std::string last_content_raw_;
    std::uint64_t replay_count_ = 0;
};

}  // namespace jointledger

#endif  // JOINT_LEDGER_PIPELINE_STAGE_HH
