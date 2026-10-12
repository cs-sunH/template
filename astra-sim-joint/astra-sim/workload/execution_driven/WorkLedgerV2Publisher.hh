#ifndef ASTRASIM_WORKLOAD_EXECUTION_DRIVEN_WORK_LEDGER_V2_PUBLISHER_HH
#define ASTRASIM_WORKLOAD_EXECUTION_DRIVEN_WORK_LEDGER_V2_PUBLISHER_HH

#include <json/json.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace AstraSim {
namespace ExecutionDriven {

/// One callback-ordered protocol-v4 transition sequencer and bounded frame
/// publisher. Both Online and execution fixtures use this same implementation.
class WorkLedgerV2Publisher {
    friend struct WorkLedgerV2PublisherTestAccess;

  public:
    explicit WorkLedgerV2Publisher(std::string epoch);

    void append_node_transition(const std::string& kind, uint64_t tick,
                                std::string work_id,
                                nlohmann::json payload);
    void append_pipeline_transition(nlohmann::json transition);
    void append_descriptor_commit(nlohmann::json declaration);

    /// Seal one sparse frame. Cursor patches are the final changed run rows
    /// for this delivery; active_resources is the one current O(A) resource
    /// frame required by pricing.
    nlohmann::json publish(uint64_t delivery_sequence, uint64_t tick,
                           nlohmann::json cursor_patches,
                           nlohmann::json active_resources);

    uint64_t next_logical_sequence() const { return next_logical_seq_; }
    uint64_t watermark() const { return delivered_watermark_; }
    uint64_t state_version() const { return state_version_; }
    bool initialized() const { return has_previous_frame_; }

  private:
    using TargetSlotKey =
        std::tuple<std::string, std::string, int,
                   std::optional<uint64_t>, size_t>;

    static void require_array(const nlohmann::json& value,
                              const char* field);
    uint64_t next_sequence_candidate() const;

    std::string epoch_;
    uint64_t next_logical_seq_ = 1;
    uint64_t delivered_watermark_ = 0;
    uint64_t state_version_ = 0;
    uint64_t last_delivery_sequence_ = 0;
    uint64_t last_tick_ = 0;
    bool has_previous_frame_ = false;
    nlohmann::json node_transitions_ = nlohmann::json::array();
    nlohmann::json pipeline_transition_runs_ = nlohmann::json::array();
    nlohmann::json descriptor_commits_ = nlohmann::json::array();
    nlohmann::json previous_resources_ = nlohmann::json::array();
    std::map<TargetSlotKey, uint64_t> target_delta_predecessor_seq_;
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // ASTRASIM_WORKLOAD_EXECUTION_DRIVEN_WORK_LEDGER_V2_PUBLISHER_HH
