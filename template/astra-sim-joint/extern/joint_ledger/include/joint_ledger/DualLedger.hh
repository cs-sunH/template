// DualLedger.hh -- orchestrator owning both ledger cores and the C API
// surface (parse -> gate -> stage -> commit -> query), plus phase timing.
#ifndef JOINT_LEDGER_DUAL_LEDGER_HH
#define JOINT_LEDGER_DUAL_LEDGER_HH

#include <memory>
#include <string>

#include "joint_ledger/PipelineRows.hh"
#include "joint_ledger/PipelineStage.hh"
#include "joint_ledger/WorkLedgerCore.hh"

namespace jointledger {

// Defined in PipelineStage.cc: routes the stage core's coordinated node
// staging (tx_stage_event/tx_shadow_work/tx_finish) to a WorkLedgerCore.
void bind_work_ledger(PipelineLedgerCore& pipeline, WorkLedgerCore& work);

class DualLedger {
 public:
    DualLedger();

    // Phase timings of the last prepared delivery (nanoseconds).
    struct PhaseNs {
        std::int64_t parse = 0;
        std::int64_t stage = 0;
        std::int64_t commit = 0;
    };
    const PhaseNs& last_phase_ns() const { return phase_ns_; }
    bool last_snapshot_full() const { return last_snapshot_full_; }

    // jl_prepare: parse + validate + stage.  Returns the prepared slot id
    // (also stored as pending_ for jl_commit).
    int prepare(const char* segment, std::size_t length,
                std::uint64_t expected_delivery, std::uint64_t expected_tick,
                std::uint64_t* event_count, std::uint64_t* seq_end,
                std::uint64_t* result_state_version);
    int commit(int prepared);
    void discard(int prepared);

    WorkLedgerCore& work() { return work_; }
    std::uint64_t active_work_count() const;
    PipelineLedgerCore& pipeline() { return pipeline_; }
    const WorkLedgerCore& work() const { return work_; }
    const PipelineLedgerCore& pipeline() const { return pipeline_; }

 private:
    WorkLedgerCore work_;
    PipelineLedgerCore pipeline_;
    std::unique_ptr<ParsedEnvelope> pending_parsed_;
    std::unique_ptr<WorkLedgerCore::Transaction> pending_tx_;
    std::string pending_segment_;
    int pending_slot_ = -1;
    PhaseNs phase_ns_;
    bool last_snapshot_full_ = false;
};

}  // namespace jointledger

#endif  // JOINT_LEDGER_DUAL_LEDGER_HH
