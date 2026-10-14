#ifndef ASTRASIM_WORKLOAD_EXECUTION_DRIVEN_WORK_LEDGER_V2_PUBLISHER_HH
#define ASTRASIM_WORKLOAD_EXECUTION_DRIVEN_WORK_LEDGER_V2_PUBLISHER_HH

#include <json/json.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "joint_ledger/CompactWire.hh"
#include "joint_ledger/TypedRows.hh"

namespace AstraSim {
namespace ExecutionDriven {

/// One published compact frame plus the header facts the bridge timing
/// surfaces consume (the encoded bytes are deterministic for identical
/// logical content, keeping the idempotent-replay raw-bytes gate exact).
struct WorkLedgerWire {
    std::string bytes;
    bool full = false;
    uint64_t delivery_sequence = 0;
    uint64_t tick = 0;
    uint64_t state_version = 0;
    uint64_t logical_seq_start = 0;
    uint64_t logical_seq_end = 0;
    uint64_t logical_event_count = 0;
};

/// One callback-ordered protocol-v4 transition sequencer and bounded frame
/// publisher. Both Online and execution fixtures use this same implementation.
/// v6: the callback buffers are typed records (no whole-row JSON DOM exists
/// on the producer path) and publish() emits workledger-compact-v1 bytes.
class WorkLedgerV2Publisher {
    friend struct WorkLedgerV2PublisherTestAccess;

  public:
    explicit WorkLedgerV2Publisher(std::string epoch);

    void append_node_transition(const std::string& kind, uint64_t tick,
                                std::string work_id,
                                nlohmann::json payload);

    /// Seal one sparse frame. active_resources is the one current O(A)
    /// resource frame required by pricing. (D11: pipeline runs, descriptor
    /// commits and cursor patches are no longer produced -- those sections
    /// stay empty on every frame and a zero-row frame is legal.)
    WorkLedgerWire publish(uint64_t delivery_sequence, uint64_t tick,
                           std::vector<jointledger::ResourceRow>
                               active_resources);

    uint64_t next_logical_sequence() const { return next_logical_seq_; }
    uint64_t watermark() const { return delivered_watermark_; }
    uint64_t state_version() const { return state_version_; }
    bool initialized() const { return has_previous_frame_; }

  private:
    static void require_nonempty(const std::string& what);
    uint64_t next_sequence_candidate() const;

    std::string epoch_;
    uint64_t next_logical_seq_ = 1;
    uint64_t delivered_watermark_ = 0;
    uint64_t state_version_ = 0;
    uint64_t last_delivery_sequence_ = 0;
    uint64_t last_tick_ = 0;
    bool has_previous_frame_ = false;
    std::vector<jointledger::CompactNodeRecord> node_transitions_;
    std::vector<jointledger::ResourceRow> previous_resources_;
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // ASTRASIM_WORKLOAD_EXECUTION_DRIVEN_WORK_LEDGER_V2_PUBLISHER_HH
