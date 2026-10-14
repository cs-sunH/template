// LedgerTypes.hh -- shared value types and strict wire readers for the
// native joint dual-ledger receiver (joint v4 acceleration round, 2026-10-12).
//
// Semantics are ported 1:1 from sh_test_mesh/workload/llama2_7b_inference/
// joint/work_ledger.py and joint/pipeline_ledger.py.  Python's json.loads
// type lattice maps onto nlohmann events as: boolean -> bool, integer ->
// number_integer/number_unsigned (Python int), real -> number_float
// (Python float).  Every reader below rejects the coercions the Python
// validators reject (bool-as-int, float-as-int, string numbers).
//
// This header is the frozen interface contract for the parallel port units;
// do not change signatures without the integrator's sign-off.
#ifndef JOINT_LEDGER_LEDGER_TYPES_HH
#define JOINT_LEDGER_LEDGER_TYPES_HH

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "json/json.hpp"

namespace jointledger {

using Json = nlohmann::json;  // json/json.hpp (vendored)

constexpr std::uint64_t U64_MAX = ~static_cast<std::uint64_t>(0);
constexpr std::uint64_t I32_MAX = (1ULL << 31) - 1;

struct LedgerError : std::runtime_error {
    explicit LedgerError(const std::string& what) : std::runtime_error(what) {}
};

// ---------------------------------------------------------------- amounts --

// WorkAmount mirrors work_ledger.WorkAmount: three optional dimensions,
// each stored as double exactly like Python's ``float(value)`` conversion in
// ``_amount_number`` (ints are converted to float at validation time).
struct WorkAmount {
    std::optional<double> ops;
    std::optional<double> bytes;
    std::optional<double> latency_ns;

    bool operator==(const WorkAmount& other) const {
        return ops == other.ops && bytes == other.bytes
            && latency_ns == other.latency_ns;
    }
    bool is_empty() const { return !ops && !bytes && !latency_ns; }
};

// ------------------------------------------------------------------ works --

// States of the work lifecycle machine (work_ledger.STATES).
enum class WorkState {
    Committed, Ready, Issued, InService, Completed, Acknowledged, Released
};
const char* work_state_name(WorkState state);          // WorkLedgerCore.cc
WorkState work_state_from_name(const std::string& name);  // throws

// work_ledger.WorkIdentity field subset used by the ledger application path.
struct WorkIdentity {
    std::string request_id;
    std::string stage;
    std::uint64_t generation = 0;
    std::uint64_t rank = 0;
    std::uint64_t node_id = 0;
    std::string operation;    // "" when absent on the wire
};

// work_ledger.CommittedWork: mutable state-machine row owned by WorkLedger.
struct CommittedWork {
    WorkIdentity identity;
    WorkState state = WorkState::Committed;
    WorkAmount total;
    WorkAmount served;
    WorkAmount remaining;
    std::string progress_source = "stale";
    std::string service_unit = "job";
    std::vector<std::string> dependencies;
    std::vector<std::string> release_conditions;
    std::optional<std::vector<std::uint64_t>> route;
    int weight = 0;
    std::optional<std::string> leg;         // nullopt = node-level work
    std::optional<std::uint64_t> committed_tick;
    std::optional<std::uint64_t> issued_tick;
    std::optional<std::uint64_t> completed_tick;
    std::optional<std::uint64_t> acked_tick;
    std::optional<std::uint64_t> released_tick;
    std::optional<std::uint64_t> as_of_tick;

    std::string work_id() const;   // canonical (or leg-suffixed) work id
    void transition(WorkState target, std::uint64_t tick);  // throws
    void apply_progress(const WorkAmount& delta, std::uint64_t tick,
                        const std::string& progress_source);  // throws
    void finish(std::uint64_t tick);
};

std::string canonical_work_id(const std::string& request_id,
                              const std::string& stage,
                              std::uint64_t generation,
                              std::uint64_t rank, std::uint64_t node_id);
std::string leg_work_id(const std::string& work_id, const std::string& leg);

// --------------------------------------------------------------- wire read --

// Strict readers over a wire JSON value; ``path`` is used in error text.
// Each mirrors the same-named Python validator in pipeline_ledger.py.
std::uint64_t read_uint(const Json& value, const std::string& path,
                        std::uint64_t maximum = U64_MAX);
std::uint64_t read_rank(const Json& value, const std::string& path);
double read_finite_nonnegative(const Json& value, const std::string& path);
std::string read_string(const Json& value, const std::string& path,
                        bool allow_empty = false);
bool read_bool(const Json& value, const std::string& path);
std::optional<std::uint64_t> read_nullable_uint(const Json& value,
                                                const std::string& path);
WorkAmount read_amount(const Json& value, const std::string& path);

// Python ``_typed_equal``: structural equality that never conflates
// bool/int/float (nlohmann signed and unsigned integers are both Python int).
bool typed_equal(const Json& left, const Json& right);

// Python ``_mapping`` with exact key set: returns the object, throws when the
// key set does not match exactly (missing/unknown).
const Json& read_mapping(const Json& value, const std::string& path);
void require_keys(const Json& object, const std::string& path,
                  const std::vector<const char*>& allowed,
                  const std::vector<const char*>& required);

// ------------------------------------------------------------- node events --

// Typed form of work_ledger._TypedLedgerEvent for the v3/v4 node stream.
// ``payload`` keeps the whitelist-validated body as a JSON object exactly as
// the Python event does (the application core reads fields by name).
struct NodeEvent {
    std::string kind;          // commit/issue/progress/completion/ack/release
    std::string work_id;
    std::uint64_t tick = 0;
    std::string producer;      // "cpp_online"
    std::uint64_t delivery_sequence = 0;
    std::uint64_t local_seq = 0;
    std::optional<WorkAmount> delta_served;
    Json payload;              // object; validated against EVENT_PAYLOAD_WHITELIST
};

}  // namespace jointledger

#endif  // JOINT_LEDGER_LEDGER_TYPES_HH
