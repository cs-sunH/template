// WorkLedgerCore.hh -- native port of the causal WorkLedger application
// core (joint/work_ledger.py), production protocol-v3/v4 transaction path.
//
// Only the coordinated transaction surface used by the online receiver is
// ported: begin_protocol_v3_transaction / tx_stage_event / tx_finish /
// tx_commit (work_ledger._WorkLedgerV2Transaction).  The legacy
// apply_payload(compact/events) path is a test/offline-only surface and
// stays on the Python reference decoder.
//
// Atomicity: a transaction stages into private overlays (works / dependents /
// retained obligations); nothing is visible through the query surface until
// tx_commit, and tx_commit only publishes already-validated rows (it cannot
// throw for content reasons).
#ifndef JOINT_LEDGER_WORK_LEDGER_CORE_HH
#define JOINT_LEDGER_WORK_LEDGER_CORE_HH

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "LedgerTypes.hh"

namespace jointledger {

class WorkLedgerCore {
 public:
    // Transaction handle; one live transaction at a time (strict one
    // in-flight delivery protocol mirrors the bridge contract).
    struct Transaction {
        friend class WorkLedgerCore;
        bool finished_committed = false;
        std::uint64_t delivery_sequence = 0;
        std::uint64_t tick = 0;
        std::string snapshot_kind;   // "full" | "delta"
        std::uint64_t previous_seq = 0;
        std::int64_t previous_tick = -1;
        std::uint64_t logical_seq_end = 0;
        bool finish_called = false;
        std::uint64_t event_count = 0;
        std::vector<std::string> touched_work_ids;
        std::unordered_set<std::string> touched_set;
        // COW overlays over the ledger's works/dependents/obligations.
        std::unordered_map<std::string, CommittedWork> work_writes;
        std::unordered_set<std::string> work_deleted;
        std::unordered_map<std::string, std::unordered_set<std::string>>
            dependents_writes;
        std::unordered_set<std::string> dependents_deleted;
        std::unordered_map<std::string, std::vector<std::string>>
            obligations_writes;
        std::unordered_set<std::string> obligations_deleted;
    };

    WorkLedgerCore();

    // ---- coordinated v3 transaction (work_ledger.py 1:1) ----
    // Throws LedgerError on any gate/semantic violation; state untouched.
    Transaction begin_protocol_v3_transaction(std::uint64_t delivery_sequence,
                                              std::uint64_t tick,
                                              const std::string& snapshot_kind);
    void tx_stage_event(Transaction& tx, std::uint64_t seq,
                        const NodeEvent& event);
    void tx_finish(Transaction& tx, std::uint64_t logical_seq_end);
    void tx_commit(Transaction& tx);   // publishes; updates counters/version

    // Read access into the transaction's staged (shadow) view; used by the
    // pipeline staging core to observe pre-event rows (mirrors
    // work_transaction.shadow._works lookups in pipeline_ledger.py).
    const CommittedWork* tx_shadow_work(const Transaction& tx,
                                        const std::string& work_id) const;

    // ---- query surface (Python-compatible JSON shapes) ----
    // Pricing frontier: {"pricing_rows": [...], "committed_not_issued": N}
    // where each row carries work_id/state/identity/total/served/remaining/
    // progress_source/service_unit/dependencies/release_conditions/route/
    // weight/leg/committed_tick/issued_tick/completed_tick/acked_tick/
    // released_tick/as_of_tick.  Rows are legs only (hbm:/pool:/noc:),
    // issued-or-later, in projection insertion order (Python _projection_order).
    Json pricing_json() const;
    // conservation_report() shape from work_ledger.py.
    Json conservation_json() const;
    // export_state() shape from work_ledger.py (nodes/hbm/pool/noc/slots).
    Json works_export_json() const;

    // ---- counters/gates (Python attribute surface) ----
    std::uint64_t version() const { return version_; }
    std::uint64_t event_watermark() const { return event_watermark_; }
    std::int64_t last_tick() const { return last_tick_; }
    std::int64_t last_delivery_sequence() const { return last_delivery_; }
    std::uint64_t full_load_count() const { return full_load_count_; }
    std::uint64_t delta_load_count() const { return delta_load_count_; }
    std::uint64_t applied_event_count() const { return applied_event_count_; }
    std::uint64_t replayed_event_count() const { return replayed_event_count_; }

    // Test/diagnostic helpers.
    const CommittedWork* work(const std::string& work_id) const;
    std::size_t active_work_count() const;

 private:
    friend struct Transaction;
    // Application core (work_ledger._apply_event_semantics 1:1); operates on
    // the transaction overlay view.
    void apply_event_semantics_(Transaction& tx, const NodeEvent& event);
    const CommittedWork* base_work_(const std::string& id) const;
    CommittedWork* overlay_work_(Transaction& tx, const std::string& id);

    // Persistent accepted state.
    std::unordered_map<std::string, CommittedWork> works_;
    std::vector<std::string> works_order_;      // insertion order (Python dict)
    std::unordered_set<std::string> works_dead_;
    std::unordered_map<std::string, std::unordered_set<std::string>>
        dependents_by_work_;
    std::unordered_map<std::string, std::vector<std::string>>
        retained_obligations_;
    std::uint64_t version_ = 0;
    std::uint64_t event_watermark_ = 0;
    std::int64_t last_tick_ = -1;
    std::int64_t last_delivery_ = -1;
    std::uint64_t applied_event_count_ = 0;
    std::uint64_t replayed_event_count_ = 0;
    std::uint64_t full_load_count_ = 0;
    std::uint64_t delta_load_count_ = 0;
    bool has_transaction_ = false;
};

}  // namespace jointledger

#endif  // JOINT_LEDGER_WORK_LEDGER_CORE_HH
