// DualLedger.cc -- delivery orchestration over the two ledger cores.
//
// Mirrors online_scheduler_base._apply_work_ledger: validate (typed parse)
// -> begin protocol-v3 WorkLedger transaction -> pipeline prepare (gate +
// global merge staging, driving tx_stage_event at each node event's exact
// logical position) -> work commit -> pipeline commit -> notify counters.
// Any throw leaves both ledgers untouched (pipeline prepare stages into COW
// overlays; work transaction stages into its own overlays; commit only
// publishes fully validated state).
#include "joint_ledger/DualLedger.hh"

#include <chrono>
#include <utility>

#include "joint_ledger/LedgerTypes.hh"

namespace jointledger {
namespace {
std::int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

DualLedger::DualLedger() {
    // N5 stage core routes coordinated node-event staging through the bound
    // WorkLedger transaction handle (bind_work_ledger lives in
    // PipelineStage.cc); bind once for the receiver's lifetime.
    bind_work_ledger(pipeline_, work_);
}
// N5 stage core routes node-event staging through the bound WorkLedger
// transaction (see PipelineStage.cc bind_work_ledger).

std::uint64_t DualLedger::active_work_count() const {
    return work_.active_work_count();
}

int DualLedger::prepare(const char* segment, std::size_t length,
                        std::uint64_t expected_delivery,
                        std::uint64_t expected_tick,
                        std::uint64_t* event_count, std::uint64_t* seq_end,
                        std::uint64_t* result_state_version) {
    if (pending_slot_ >= 0) {
        throw LedgerError("previous prepared delivery was not committed");
    }
    const std::int64_t t0 = now_ns();
    RetainedRegistry retained;
    retained.descriptors = &pipeline_.descriptors();
    retained.selector_indexes = &pipeline_.selector_indexes();
    // validate_envelope's retained-replay allowance: a delivery with the
    // same sequence as the last accepted one is an idempotent replay gate.
    const bool allow_retained_replay =
        static_cast<std::int64_t>(expected_delivery)
            == pipeline_.delivery_sequence()
        && pipeline_.delivery_sequence() >= 0;
    auto parsed = std::make_unique<ParsedEnvelope>(parse_envelope_v4(
        segment, length, retained, allow_retained_replay,
        expected_delivery, expected_tick));
    const std::int64_t t1 = now_ns();

    auto tx = std::make_unique<WorkLedgerCore::Transaction>(
        work_.begin_protocol_v3_transaction(
            parsed->header.delivery_sequence, parsed->header.tick,
            parsed->header.full ? "full" : "delta"));
    const std::string segment_text(segment, length);
    const int slot = pipeline_.prepare(*parsed, *tx, segment_text);
    const std::int64_t t2 = now_ns();

    if (event_count != nullptr) *event_count = parsed->header.event_count;
    if (seq_end != nullptr) *seq_end = parsed->header.seq_end;
    if (result_state_version != nullptr) {
        *result_state_version = parsed->header.result_version;
    }
    last_snapshot_full_ = parsed->header.full;
    pending_parsed_ = std::move(parsed);
    pending_tx_ = std::move(tx);
    pending_segment_ = std::move(segment_text);
    pending_slot_ = slot;
    phase_ns_ = PhaseNs{t1 - t0, t2 - t1, 0};
    return slot;
}

int DualLedger::commit(int prepared) {
    if (prepared != pending_slot_ || pending_slot_ < 0) {
        throw LedgerError("unknown prepared delivery slot");
    }
    const std::int64_t t0 = now_ns();
    // WorkLedger first: its commit mutates touched rows only and cannot
    // reject after preparation (online_scheduler_base ordering).
    work_.tx_commit(*pending_tx_);
    pipeline_.commit_prepared(pending_slot_);
    const std::int64_t t1 = now_ns();
    phase_ns_.commit = t1 - t0;
    pending_parsed_.reset();
    pending_tx_.reset();
    pending_segment_.clear();
    pending_slot_ = -1;
    return 0;
}

void DualLedger::discard(int prepared) {
    if (prepared != pending_slot_ || pending_slot_ < 0) {
        throw LedgerError("unknown prepared delivery slot");
    }
    if (pending_slot_ >= 0) {
        pipeline_.discard_prepared(pending_slot_);
    }
    pending_parsed_.reset();
    pending_tx_.reset();
    pending_segment_.clear();
    pending_slot_ = -1;
}

}  // namespace jointledger
