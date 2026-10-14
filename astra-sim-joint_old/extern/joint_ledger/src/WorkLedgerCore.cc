// WorkLedgerCore.cc -- strict port of the causal WorkLedger application
// core (sh_test_mesh/workload/llama2_7b_inference/joint/work_ledger.py),
// production protocol-v3/v4 coordinated transaction path only:
//
//   * WorkLedger.begin_protocol_v3_transaction        (work_ledger.py 918-945)
//   * _WorkLedgerV2Transaction.stage/finish/commit    (work_ledger.py 1846-1956)
//   * WorkLedger._apply_event_semantics               (work_ledger.py 1418-1497)
//   * WorkLedger._release_work                        (work_ledger.py 1499-1534)
//   * WorkLedger.export_state                         (work_ledger.py 1556-1718)
//   * WorkLedger.conservation_report                  (work_ledger.py 1728-1774)
//   * _work_from_event                                (work_ledger.py 2140-2187)
//   * _work_from_commit_parts                         (work_ledger.py 2227-2310)
//   * _split_work_id/_request_of/_stage_of/
//     _generation_of/_rank_of/_node_of                (work_ledger.py 2313-2353)
//
// The legacy apply_payload(compact/events) decoder and the full-snapshot
// parser (parse_work_ledger_state) are offline/test-only Python surfaces and
// are intentionally not ported (see WorkLedgerCore.hh).
#include "joint_ledger/WorkLedgerCore.hh"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>

namespace jointledger {
namespace {

// work_ledger.AMOUNT_EPSILON (1e-6) and the relative conservation term
// (1e-9*|total|): tolerance arithmetic copied verbatim.
constexpr double kAmountEpsilon = 1e-6;
constexpr double kRelativeTolerance = 1e-9;

const char* const kDimensionNames[3] = {"ops", "bytes", "latency_ns"};

const std::optional<double>& amount_dim(const WorkAmount& amount, int dim) {
    return dim == 0 ? amount.ops : dim == 1 ? amount.bytes : amount.latency_ns;
}

std::optional<double>& amount_dim(WorkAmount& amount, int dim) {
    return dim == 0 ? amount.ops : dim == 1 ? amount.bytes : amount.latency_ns;
}

// work_ledger._amount_number: decode one wire amount without accepting
// coercions (bool) or non-finite data.  Negative values are *not* rejected
// here; WorkAmount.validate() owns the non-negativity domain.
double amount_number_(const Json& value, const std::string& field_name) {
    if (value.is_boolean()
            || !(value.is_number_integer() || value.is_number_unsigned()
                 || value.is_number_float())) {
        throw LedgerError("work amount '" + field_name
                          + "' must be a finite number");
    }
    const double number = value.get<double>();
    if (!std::isfinite(number)) {
        throw LedgerError("work amount '" + field_name + "' must be finite");
    }
    return number;
}

// work_ledger.WorkAmount.validate(): finite and non-negative within epsilon.
void validate_amount_(const WorkAmount& amount) {
    for (int dim = 0; dim < 3; ++dim) {
        const std::optional<double>& value = amount_dim(amount, dim);
        if (!value) continue;
        if (!std::isfinite(*value) || *value < -kAmountEpsilon) {
            throw LedgerError(std::string("work amount '")
                              + kDimensionNames[dim]
                              + "' must be finite and non-negative");
        }
    }
}

// work_ledger._opt_sub (the add half of the pair lives inside
// CommittedWork::apply_progress in LedgerTypes.cc).
std::optional<double> opt_sub_(const std::optional<double>& a,
                               const std::optional<double>& b) {
    if (!a || !b) return a;
    return *a - *b;
}

// work_ledger.WorkAmount.minus (validated arithmetic; plus is unused here
// because CommittedWork::apply_progress already implements it).
WorkAmount amount_minus_(const WorkAmount& a, const WorkAmount& b) {
    WorkAmount out{opt_sub_(a.ops, b.ops), opt_sub_(a.bytes, b.bytes),
                   opt_sub_(a.latency_ns, b.latency_ns)};
    validate_amount_(out);
    return out;
}

// work_ledger.amount_from_mapping: missing dimension keys decode as None,
// unknown keys fail closed (reported sorted like Python's sorted(list)).
WorkAmount amount_from_mapping_(const Json& value) {
    if (value.is_null()) return WorkAmount{};
    if (!value.is_object()) {
        throw LedgerError("work amount must be a mapping");
    }
    std::vector<std::string> unknown;
    for (auto member = value.begin(); member != value.end(); ++member) {
        if (member.key() != "ops" && member.key() != "bytes"
                && member.key() != "latency_ns") {
            unknown.push_back(member.key());
        }
    }
    if (!unknown.empty()) {
        std::sort(unknown.begin(), unknown.end());
        std::string joined;
        for (const std::string& key : unknown) {
            if (!joined.empty()) joined += ", ";
            joined += "'" + key + "'";
        }
        throw LedgerError("unknown work-amount keys [" + joined + "]");
    }
    WorkAmount out;
    for (int dim = 0; dim < 3; ++dim) {
        const auto member = value.find(kDimensionNames[dim]);
        if (member == value.end() || member->is_null()) continue;
        amount_dim(out, dim) = amount_number_(*member, kDimensionNames[dim]);
    }
    validate_amount_(out);
    return out;
}

// Python int() coercion over the JSON value lattice: bool -> 0/1, integer ->
// exact, float -> truncation toward zero, "123" -> 123.  null/objects/arrays
// raise TypeError in Python; the native core fails closed with LedgerError.
std::int64_t py_int_(const Json& value, const std::string& path) {
    if (value.is_boolean()) return value.get<bool>() ? 1 : 0;
    if (value.is_number_unsigned()) {
        const std::uint64_t raw = value.get<std::uint64_t>();
        if (raw > 9223372036854775807ULL) {
            throw LedgerError(path + " exceeds the native int64 range");
        }
        return static_cast<std::int64_t>(raw);
    }
    if (value.is_number_integer()) return value.get<std::int64_t>();
    if (value.is_number_float()) {
        const double number = value.get<double>();
        if (!std::isfinite(number) || number >= 9.3e18
                || number <= -9.3e18) {
            throw LedgerError(path + " exceeds the native int64 range");
        }
        return static_cast<std::int64_t>(number);
    }
    if (value.is_string()) {
        const std::string& text = value.get_ref<const std::string&>();
        std::size_t begin = 0;
        const std::size_t end = text.size();
        bool negative = false;
        if (begin < end && (text[begin] == '+' || text[begin] == '-')) {
            negative = text[begin] == '-';
            ++begin;
        }
        if (begin >= end) throw LedgerError(path + " must be an integer");
        unsigned long long accumulated = 0;
        for (std::size_t i = begin; i < end; ++i) {
            if (text[i] < '0' || text[i] > '9') {
                throw LedgerError(path + " must be an integer");
            }
            accumulated = accumulated * 10u
                + static_cast<unsigned long long>(text[i] - '0');
            if (accumulated > 9223372036854775807ULL) {
                throw LedgerError(path + " exceeds the native int64 range");
            }
        }
        return negative ? -static_cast<std::int64_t>(accumulated)
                        : static_cast<std::int64_t>(accumulated);
    }
    throw LedgerError(path + " must be an integer");
}

// Python int(str) for work_id segments and pool leg parts; parse failures map
// to the caller's fallback (work_ledger catches and returns 0 / fails).
bool parse_int_segment_(const std::string& text, std::int64_t* out) {
    std::size_t begin = 0;
    bool negative = false;
    if (begin < text.size() && (text[begin] == '+' || text[begin] == '-')) {
        negative = text[begin] == '-';
        ++begin;
    }
    if (begin >= text.size()) return false;
    unsigned long long accumulated = 0;
    for (std::size_t i = begin; i < text.size(); ++i) {
        if (text[i] < '0' || text[i] > '9') return false;
        accumulated = accumulated * 10u
            + static_cast<unsigned long long>(text[i] - '0');
        if (accumulated > 9223372036854775807ULL) return false;
    }
    *out = negative ? -static_cast<std::int64_t>(accumulated)
                    : static_cast<std::int64_t>(accumulated);
    return true;
}

// work_ledger._split_work_id -> [request, stage, generation, uid].
struct SplitWorkId {
    std::string request;
    std::string stage;
    std::string generation;
    std::string uid;
};

SplitWorkId split_work_id_(const std::string& work_id) {
    const std::size_t at = work_id.find('@');
    const std::string node =
        at == std::string::npos ? work_id : work_id.substr(0, at);
    std::string body;
    std::string uid;
    const std::size_t colon = node.rfind(':');
    if (colon == std::string::npos) {
        uid = node;  // rpartition(":") of a separator-less node
    } else {
        body = node.substr(0, colon);
        uid = node.substr(colon + 1);
    }
    std::string request = body;
    std::string rest;
    const std::size_t hash = body.find('#');
    if (hash != std::string::npos) {
        request = body.substr(0, hash);
        rest = body.substr(hash + 1);
    }
    std::string stage = rest;
    std::string generation;
    const std::size_t stage_colon = rest.find(':');
    if (stage_colon != std::string::npos) {
        stage = rest.substr(0, stage_colon);
        generation = rest.substr(stage_colon + 1);
    }
    return SplitWorkId{request, stage, generation, uid};
}

// work_ledger._request_of / _stage_of / _generation_of / _rank_of /
// _node_of: non-canonical shapes (NoC legs) fall back like the Python
// IndexError/ValueError handlers.
std::string request_of_(const std::string& work_id) {
    return split_work_id_(work_id).request;
}

std::string stage_of_(const std::string& work_id) {
    return split_work_id_(work_id).stage;
}

std::uint64_t generation_of_(const std::string& work_id) {
    std::int64_t parsed = 0;
    if (!parse_int_segment_(split_work_id_(work_id).generation, &parsed)
            || parsed < 0) {
        return 0;
    }
    return static_cast<std::uint64_t>(parsed);
}

std::uint64_t rank_of_(const std::string& work_id) {
    const std::string& uid = split_work_id_(work_id).uid;
    if (uid.size() < 2) return 0;
    const std::size_t node_mark = uid.find('n', 1);
    const std::string rank_text =
        node_mark == std::string::npos ? uid.substr(1)
                                       : uid.substr(1, node_mark - 1);
    std::int64_t parsed = 0;
    if (!parse_int_segment_(rank_text, &parsed) || parsed < 0) return 0;
    return static_cast<std::uint64_t>(parsed);
}

std::uint64_t node_of_(const std::string& work_id) {
    const std::string& uid = split_work_id_(work_id).uid;
    const std::size_t node_mark = uid.find('n');
    if (node_mark == std::string::npos) return 0;
    std::int64_t parsed = 0;
    if (!parse_int_segment_(uid.substr(node_mark + 1), &parsed) || parsed < 0) {
        return 0;
    }
    return static_cast<std::uint64_t>(parsed);
}

// work_ledger._LEG_COMMIT_STATE_NAMES (keys are Python str.lower() forms).
WorkState leg_commit_state_(const std::string& lowered) {
    if (lowered == "active") return WorkState::InService;
    if (lowered == "propagatingtail") return WorkState::Completed;
    if (lowered == "streaming") return WorkState::InService;
    if (lowered == "latencywaiting") return WorkState::Issued;
    if (lowered == "completed") return WorkState::Completed;
    if (lowered == "fluidcompleteawaitingcallback") return WorkState::Completed;
    throw LedgerError("unknown commit state '" + lowered + "'");
}

std::string ascii_lower_(const std::string& text) {
    std::string out = text;
    for (char& c : out) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    return out;
}

// work_ledger._normalized_commit_state: state-machine names pass through,
// legacy C++ resource-leg lifecycle strings map, unknown names fail closed.
WorkState normalized_commit_state_(const Json& state) {
    if (!state.is_string()) {
        throw LedgerError("commit state must be a string");
    }
    const std::string name = state.get<std::string>();
    if (name == "committed") return WorkState::Committed;
    if (name == "ready") return WorkState::Ready;
    if (name == "issued") return WorkState::Issued;
    if (name == "in_service") return WorkState::InService;
    if (name == "completed") return WorkState::Completed;
    if (name == "acknowledged") return WorkState::Acknowledged;
    if (name == "released") return WorkState::Released;
    return leg_commit_state_(ascii_lower_(name));
}

// Strict string field read.  Python stores the raw wire value inside
// WorkIdentity and stringifies lazily at format time; the native core fails
// closed on non-string values instead (documented deviation; production
// frames always carry strings).
std::string string_field_(const Json& payload, const char* key,
                          const std::string& fallback) {
    const auto member = payload.find(key);
    if (member == payload.end()) return fallback;
    if (!member->is_string()) {
        throw LedgerError(std::string("work ledger field '") + key
                          + "' must be a string");
    }
    return member->get<std::string>();
}

std::vector<std::string> string_list_field_(const Json& payload,
                                            const char* key) {
    const auto member = payload.find(key);
    if (member == payload.end()) return {};
    if (!member->is_array()) {
        throw LedgerError(std::string("work ledger field '") + key
                          + "' must be an array of strings");
    }
    std::vector<std::string> out;
    out.reserve(member->size());
    for (const Json& element : *member) {
        if (!element.is_string()) {
            throw LedgerError(std::string("work ledger field '") + key
                              + "' must be an array of strings");
        }
        out.push_back(element.get<std::string>());
    }
    return out;
}

// Python tuple(x for x in value) with per-element int() coercion (parents,
// route); bools coerce to 0/1 exactly like Python int().
std::vector<std::uint64_t> int_list_field_(const Json& payload, const char* key) {
    const std::string path = std::string("work ledger field '") + key + "'";
    const auto member = payload.find(key);
    if (member == payload.end()) return {};
    if (!member->is_array()) {
        throw LedgerError(path + " must be an array of integers");
    }
    std::vector<std::uint64_t> out;
    out.reserve(member->size());
    for (const Json& element : *member) {
        const std::int64_t parsed = py_int_(element, path + " element");
        if (parsed < 0) {
            throw LedgerError(path + " element must be non-negative");
        }
        out.push_back(static_cast<std::uint64_t>(parsed));
    }
    return out;
}

// WorkAmount.as_dict(): {"ops": .., "bytes": .., "latency_ns": ..} with null
// for inapplicable dimensions.
Json amount_json_(const WorkAmount& amount) {
    Json out = Json::object();
    for (int dim = 0; dim < 3; ++dim) {
        const std::optional<double>& value = amount_dim(amount, dim);
        out[kDimensionNames[dim]] = value ? Json(*value) : Json();
    }
    return out;
}

Json optional_tick_json_(const std::optional<std::uint64_t>& tick) {
    return tick ? Json(*tick) : Json();
}

// work_ledger._is_pricing_row: legs past the issue boundary only.
bool is_pricing_row_(const CommittedWork& work) {
    if (work.state == WorkState::Committed || work.state == WorkState::Ready) {
        return false;
    }
    if (!work.leg) return false;
    const std::string& leg = *work.leg;
    return leg.rfind("hbm:", 0) == 0 || leg.rfind("pool:", 0) == 0
        || leg.rfind("noc:", 0) == 0;
}

// work_ledger._work_from_commit_parts (2227-2310).
CommittedWork work_from_commit_parts_(
    const std::string& request_id, const std::string& stage,
    std::uint64_t generation, std::uint64_t rank, std::uint64_t node_id,
    const std::string& operation, const std::optional<std::string>& leg,
    const WorkAmount& total, const WorkAmount& remaining_input, WorkState state,
    std::vector<std::string> dependencies, const std::vector<std::uint64_t>& parents,
    std::vector<std::string> release_conditions,
    std::optional<std::vector<std::uint64_t>> route,
    const std::string& service_unit, std::uint64_t tick) {
    CommittedWork work;
    work.identity.request_id = request_id;
    work.identity.stage = stage;
    work.identity.generation = generation;
    work.identity.rank = rank;
    work.identity.node_id = node_id;
    work.identity.operation = operation;
    work.state = WorkState::Committed;
    work.total = total;
    // Explicit canonical dependencies take precedence; parents canonicalize
    // through the identity-based form (work_ledger.py 2263-2271).
    if (dependencies.empty() && !parents.empty()) {
        dependencies.clear();
        dependencies.reserve(parents.size());
        for (const std::uint64_t parent : parents) {
            dependencies.push_back(
                canonical_work_id(request_id, stage, generation, rank, parent));
        }
    }
    work.dependencies = std::move(dependencies);
    work.release_conditions = std::move(release_conditions);
    if (route && route->empty()) route.reset();  // (tuple(x) if x else None)
    work.route = std::move(route);
    work.leg = leg;
    work.committed_tick = tick;
    const bool issued_state =
        state == WorkState::Issued || state == WorkState::InService;
    if (!(remaining_input == WorkAmount{})) {
        // Mid-flight first sight with an authoritative residual: served before
        // first sight = total - remaining per dimension (missing dimensions
        // are not invented); remaining > total fails closed in the subtract.
        work.remaining = remaining_input;
        work.served = amount_minus_(total, remaining_input);
        work.progress_source = "authoritative";
    } else {
        // No authoritative residual: remaining is a total upper-bound
        // placeholder, disclosed via stale/inferred (work_ledger.py 2283-2285).
        work.remaining = total;
        work.served = WorkAmount{};
        work.progress_source =
            state == WorkState::InService ? "inferred" : "stale";
    }
    work.service_unit = service_unit;
    if (issued_state) work.transition(WorkState::Issued, tick);
    if (state == WorkState::InService) {
        work.transition(WorkState::InService, tick);
    }
    if (state == WorkState::Completed) {
        // Terminal first sight: replay to completed with served = total
        // (issue+finish+GC all happened inside one snapshot interval).
        work.transition(WorkState::Issued, tick);
        work.finish(tick);
        work.served = work.total;
    }
    return work;
}

// work_ledger._work_from_event (2140-2187): identity fields are read from the
// event payload top level (the C++ commit_payload kCarry surface).
CommittedWork work_from_event_(const NodeEvent& event) {
    const Json& payload = event.payload;
    static const char* const kRemainingKeys[3] = {
        "remaining_ops", "remaining_bytes", "remaining_latency_ns"};
    WorkAmount remaining_input;
    for (int dim = 0; dim < 3; ++dim) {
        const auto member = payload.find(kRemainingKeys[dim]);
        if (member == payload.end() || member->is_null()) continue;
        amount_dim(remaining_input, dim) =
            amount_number_(*member, kRemainingKeys[dim]);
    }
    validate_amount_(remaining_input);

    const std::string request_id =
        string_field_(payload, "request_id", request_of_(event.work_id));
    const std::string stage =
        string_field_(payload, "stage", stage_of_(event.work_id));
    std::uint64_t generation = generation_of_(event.work_id);
    std::uint64_t rank = rank_of_(event.work_id);
    std::uint64_t node_id = node_of_(event.work_id);
    if (const auto member = payload.find("generation");
        member != payload.end()) {
        const std::int64_t parsed = py_int_(*member, "work ledger generation");
        if (parsed < 0) {
            throw LedgerError("work ledger generation must be non-negative");
        }
        generation = static_cast<std::uint64_t>(parsed);
    }
    if (const auto member = payload.find("rank"); member != payload.end()) {
        const std::int64_t parsed = py_int_(*member, "work ledger rank");
        if (parsed < 0) {
            throw LedgerError("work ledger rank must be non-negative");
        }
        rank = static_cast<std::uint64_t>(parsed);
    }
    if (const auto member = payload.find("node_id"); member != payload.end()) {
        const std::int64_t parsed = py_int_(*member, "work ledger node_id");
        if (parsed < 0) {
            throw LedgerError("work ledger node_id must be non-negative");
        }
        node_id = static_cast<std::uint64_t>(parsed);
    } else if (const auto flow = payload.find("flow_id");
               flow != payload.end()) {
        const std::int64_t parsed = py_int_(*flow, "work ledger flow_id");
        if (parsed < 0) {
            throw LedgerError("work ledger flow_id must be non-negative");
        }
        node_id = static_cast<std::uint64_t>(parsed);
    }
    const std::string operation = string_field_(payload, "operation", "");
    std::optional<std::string> leg;
    if (const auto member = payload.find("leg");
        member != payload.end() && !member->is_null()) {
        if (!member->is_string()) {
            throw LedgerError("work ledger field 'leg' must be a string");
        }
        leg = member->get<std::string>();
    }
    WorkAmount total = WorkAmount{};
    if (const auto member = payload.find("total"); member != payload.end()) {
        total = amount_from_mapping_(*member);
    }
    WorkState state = WorkState::Committed;
    if (const auto member = payload.find("state"); member != payload.end()) {
        state = normalized_commit_state_(*member);
    }
    std::vector<std::string> dependencies =
        string_list_field_(payload, "dependencies");
    // Python: parents=None is falsy and skips canonicalization entirely
    // (work_ledger.py 2266: "if not dependencies and parents").
    std::vector<std::uint64_t> parents;
    if (const auto member = payload.find("parents");
        member != payload.end() && !member->is_null()) {
        parents = int_list_field_(payload, "parents");
    }
    std::vector<std::string> release_conditions =
        string_list_field_(payload, "release_conditions");
    std::optional<std::vector<std::uint64_t>> route;
    if (const auto member = payload.find("route");
        member != payload.end() && !member->is_null()) {
        if (!member->is_array()) {
            throw LedgerError("work ledger field 'route' must be an array");
        }
        route = int_list_field_(payload, "route");
    }
    const std::string service_unit =
        string_field_(payload, "service_unit", "job");
    return work_from_commit_parts_(
        request_id, stage, generation, rank, node_id, operation, leg, total,
        remaining_input, state, std::move(dependencies), parents,
        std::move(release_conditions), std::move(route), service_unit,
        event.tick);
}

}  // namespace

// ------------------------------------------------------------ construction --

WorkLedgerCore::WorkLedgerCore() = default;

const CommittedWork* WorkLedgerCore::base_work_(const std::string& id) const {
    const auto it = works_.find(id);
    return it == works_.end() ? nullptr : &it->second;
}

const CommittedWork* WorkLedgerCore::tx_shadow_work(
    const Transaction& tx, const std::string& work_id) const {
    const auto write = tx.work_writes.find(work_id);
    if (write != tx.work_writes.end()) return &write->second;
    if (tx.work_deleted.count(work_id)) return nullptr;
    return base_work_(work_id);
}

CommittedWork* WorkLedgerCore::overlay_work_(Transaction& tx,
                                             const std::string& id) {
    const auto write = tx.work_writes.find(id);
    if (write != tx.work_writes.end()) return &write->second;
    if (tx.work_deleted.count(id)) return nullptr;
    const CommittedWork* base = base_work_(id);
    if (base == nullptr) return nullptr;
    return &tx.work_writes.emplace(id, *base).first->second;
}

const CommittedWork* WorkLedgerCore::work(const std::string& work_id) const {
    return base_work_(work_id);
}

std::size_t WorkLedgerCore::active_work_count() const {
    return works_.size();
}

// ---------------------------------------------------- v3 transaction surface --

WorkLedgerCore::Transaction WorkLedgerCore::begin_protocol_v3_transaction(
    std::uint64_t delivery_sequence, std::uint64_t tick,
    const std::string& snapshot_kind) {
    if (snapshot_kind != "full" && snapshot_kind != "delta") {
        throw LedgerError("v3 snapshot_kind must be full or delta");
    }
    if (last_delivery_ >= 0
            && delivery_sequence <= static_cast<std::uint64_t>(last_delivery_)) {
        throw LedgerError("v3 delivery sequence is not monotonic");
    }
    if (static_cast<std::int64_t>(tick) < last_tick_) {
        throw LedgerError("v3 envelope tick regressed");
    }
    if (snapshot_kind == "full"
            && (!works_.empty() || full_load_count_ != 0
                || delta_load_count_ != 0)) {
        throw LedgerError("v3 full snapshot requires a pristine WorkLedger");
    }
    Transaction tx;
    tx.delivery_sequence = delivery_sequence;
    tx.tick = tick;
    tx.snapshot_kind = snapshot_kind;
    tx.previous_seq = 0;
    tx.previous_tick = last_tick_;
    has_transaction_ = true;
    return tx;
}

void WorkLedgerCore::tx_stage_event(Transaction& tx, std::uint64_t seq,
                                    const NodeEvent& event) {
    if (tx.finished_committed || tx.finish_called) {
        throw LedgerError("cannot stage events after transaction finish");
    }
    if (seq <= tx.previous_seq) {
        // seq > 0 is implied: previous_seq starts at 0 and uint64 seq cannot
        // be negative (work_ledger: seq <= previous or seq <= 0 fails).
        throw LedgerError(
            "v3 node transitions must preserve increasing logical_seq");
    }
    if (event.local_seq != seq
            || event.delivery_sequence != tx.delivery_sequence) {
        throw LedgerError(
            "v3 typed node event identity disagrees with envelope");
    }
    if (static_cast<std::int64_t>(event.tick) < tx.previous_tick
            || event.tick > tx.tick) {
        throw LedgerError(
            "v3 node event tick is outside monotonic envelope time");
    }
    apply_event_semantics_(tx, event);
    if (tx.touched_set.insert(event.work_id).second) {
        tx.touched_work_ids.push_back(event.work_id);
    }
    ++tx.event_count;
    tx.previous_seq = seq;
    tx.previous_tick = static_cast<std::int64_t>(event.tick);
}

void WorkLedgerCore::tx_finish(Transaction& tx,
                               std::uint64_t logical_seq_end) {
    if (tx.previous_seq > logical_seq_end) {
        throw LedgerError(
            "v3 logical sequence end precedes a staged node event");
    }
    tx.logical_seq_end = logical_seq_end;
    tx.finish_called = true;
}

void WorkLedgerCore::tx_commit(Transaction& tx) {
    if (tx.finished_committed) {
        throw LedgerError("v2 WorkLedger transaction already committed");
    }
    // Python retains the final staged event's canonical content here for the
    // adjacent-replay gate; canonicalization is owned by the native envelope
    // reader and the frozen NodeEvent surface carries no canonical field, so
    // that check lives outside this unit (see the delivery report).
    //
    // works_order_/works_dead_ maintain the Python dict insertion order
    // (projection order): append on insert, tombstone on delete, re-insertion
    // appends a fresh slot at the end.
    auto order_insert = [this](const std::string& id) {
        if (works_dead_.erase(id) != 0) {
            const auto stale =
                std::find(works_order_.begin(), works_order_.end(), id);
            if (stale != works_order_.end()) works_order_.erase(stale);
        }
        if (std::find(works_order_.begin(), works_order_.end(), id)
            == works_order_.end()) {
            works_order_.push_back(id);
        }
    };
    // Publish the three COW overlays.  Python's _CowOverlay.apply_to pops
    // deletions first and then updates writes in dict (write-insertion)
    // order; the write-insertion order of a transaction equals the
    // first-touch order of its staged events (touched_work_ids), because the
    // first event touching a work is exactly what creates its overlay entry.
    // Unordered-map iteration order must not leak into works_order_.
    for (const std::string& id : tx.touched_work_ids) {
        const bool deleted = tx.work_deleted.count(id) != 0;
        const auto write = tx.work_writes.find(id);
        if (deleted) {
            works_.erase(id);
            works_dead_.insert(id);
        }
        if (write != tx.work_writes.end()) {
            // A key deleted and rewritten in the same transaction re-adds at
            // the end of the dict order (Python pop-then-update).
            const bool active_before = !deleted && works_.count(id) != 0;
            works_[id] = write->second;
            if (!active_before) order_insert(id);
        }
    }
    for (const std::string& id : tx.dependents_deleted) {
        dependents_by_work_.erase(id);
    }
    for (const auto& write : tx.dependents_writes) {
        dependents_by_work_[write.first] = write.second;
    }
    for (const std::string& id : tx.obligations_deleted) {
        retained_obligations_.erase(id);
    }
    for (const auto& write : tx.obligations_writes) {
        retained_obligations_[write.first] = write.second;
    }
    applied_event_count_ += tx.event_count;
    if (tx.snapshot_kind == "full") {
        ++full_load_count_;
    } else {
        ++delta_load_count_;
    }
    event_watermark_ = tx.logical_seq_end;
    last_delivery_ = static_cast<std::int64_t>(tx.delivery_sequence);
    last_tick_ = std::max(last_tick_, static_cast<std::int64_t>(tx.tick));
    if (tx.event_count != 0) {
        ++version_;
    }
    tx.finished_committed = true;
    has_transaction_ = false;
}

// ----------------------------------------------------- event application core --

void WorkLedgerCore::apply_event_semantics_(Transaction& tx,
                                            const NodeEvent& event) {
    // Overlay read-through helpers (Python: shadow._works / shadow.
    // _dependents_by_work COW overlays with copy-on-read semantics; reads
    // never mutate observable state, so plain const reads are equivalent).
    auto view_dependents =
        [&](const std::string& key) -> const std::unordered_set<std::string>* {
        const auto write = tx.dependents_writes.find(key);
        if (write != tx.dependents_writes.end()) return &write->second;
        if (tx.dependents_deleted.count(key)) return nullptr;
        const auto base = dependents_by_work_.find(key);
        return base == dependents_by_work_.end() ? nullptr : &base->second;
    };

    const CommittedWork* staged = tx_shadow_work(tx, event.work_id);
    if (event.kind == "commit") {
        if (staged != nullptr) {
            throw LedgerError("commit event for already-known work "
                              + event.work_id);
        }
        const CommittedWork work = work_from_event_(event);
        tx.work_deleted.erase(event.work_id);
        tx.work_writes[event.work_id] = work;
        for (const std::string& dependency : work.dependencies) {
            // setdefault(dependency, set()).add(work_id)
            std::unordered_set<std::string> dependents;
            if (const std::unordered_set<std::string>* current =
                    view_dependents(dependency)) {
                dependents = *current;
            }
            dependents.insert(event.work_id);
            tx.dependents_deleted.erase(dependency);
            tx.dependents_writes[dependency] = std::move(dependents);
        }
        return;
    }
    if (staged == nullptr) {
        throw LedgerError(event.kind + " event for unknown work "
                          + event.work_id + " (full resync missing?)");
    }
    // work_ledger._KIND_SOURCE_STATES domain check (applied before dispatch).
    const WorkState state = staged->state;
    const bool source_ok =
        event.kind == "issue"
            ? (state == WorkState::Committed || state == WorkState::Ready)
        : event.kind == "progress"
            ? (state == WorkState::Issued || state == WorkState::InService
                || state == WorkState::Committed || state == WorkState::Ready)
        : event.kind == "completion"
            ? (state == WorkState::Committed || state == WorkState::Ready
                || state == WorkState::Issued || state == WorkState::InService
                || state == WorkState::Completed)
        : event.kind == "ack"
            ? state == WorkState::Completed
        : event.kind == "release"
            ? (state == WorkState::Completed
                || state == WorkState::Acknowledged
                || state == WorkState::Released)
        : true;  // unknown kinds carry no domain (reader validates the set)
    if (!source_ok) {
        throw LedgerError(event.kind + " event on work " + event.work_id
                          + " in state '" + work_state_name(state)
                          + "' (illegal source state)");
    }
    if (event.kind == "issue") {
        CommittedWork* work = overlay_work_(tx, event.work_id);
        work->transition(WorkState::Issued, event.tick);
        return;
    }
    if (event.kind == "progress") {
        CommittedWork* work = overlay_work_(tx, event.work_id);
        const WorkAmount delta =
            event.delta_served ? *event.delta_served : WorkAmount{};
        // Authoritative residual agreement check + served recomputation
        // (work_ledger.py 1446-1491).
        std::vector<std::pair<int, double>> remaining_updates;
        static const char* const kRemainingFields[3] = {
            "remaining_ops", "remaining_bytes", "remaining_latency_ns"};
        for (int dim = 0; dim < 3; ++dim) {
            const auto member = event.payload.find(kRemainingFields[dim]);
            if (member == event.payload.end()) continue;
            const double wire_remaining = amount_number_(
                *member, std::string("event.") + kRemainingFields[dim]);
            const std::optional<double> delta_value = amount_dim(delta, dim);
            const std::optional<double> current_value =
                amount_dim(work->remaining, dim);
            if (!delta_value || !current_value) {
                throw LedgerError("progress event " + event.work_id
                                  + " carries " + kRemainingFields[dim]
                                  + " without a matching remaining/delta "
                                    "dimension");
            }
            const double expected_remaining = *current_value - *delta_value;
            const double tolerance =
                kAmountEpsilon + kRelativeTolerance * std::fabs(wire_remaining);
            if (std::fabs(expected_remaining - wire_remaining) > tolerance) {
                throw LedgerError("progress event " + event.work_id + " "
                                  + kRemainingFields[dim]
                                  + " disagrees with delta: expected "
                                  + std::to_string(expected_remaining)
                                  + ", got " + std::to_string(wire_remaining));
            }
            remaining_updates.emplace_back(dim, wire_remaining);
        }
        work->apply_progress(delta, event.tick, "inferred");
        if (!remaining_updates.empty()) {
            // Preserve the backend's exact tick-end residual and recompute
            // served for the affected dimensions so the conservation line
            // stays exact after replacing the rounded subtraction.
            WorkAmount remaining = work->remaining;
            WorkAmount served = work->served;
            for (const auto& update : remaining_updates) {
                amount_dim(remaining, update.first) = update.second;
                const std::optional<double> total_value =
                    amount_dim(work->total, update.first);
                if (total_value) {
                    amount_dim(served, update.first) =
                        *total_value - update.second;
                }
            }
            validate_amount_(remaining);
            validate_amount_(served);
            work->remaining = remaining;
            work->served = served;
        }
        return;
    }
    if (event.kind == "completion") {
        CommittedWork* work = overlay_work_(tx, event.work_id);
        work->finish(event.tick);
        return;
    }
    if (event.kind == "ack") {
        CommittedWork* work = overlay_work_(tx, event.work_id);
        work->transition(WorkState::Acknowledged, event.tick);
        return;
    }
    if (event.kind == "release") {
        // ---- work_ledger._release_work (1499-1534), overlay view ----
        if (staged->state == WorkState::Released) return;  // idempotent
        CommittedWork* work = overlay_work_(tx, event.work_id);
        work->transition(WorkState::Released, event.tick);
        if (!work->release_conditions.empty()) {
            // Retired obligations stay on file while an active work still
            // references this one (work_ledger.py 1507-1517).
            bool still_depended = false;
            if (const std::unordered_set<std::string>* dependents =
                    view_dependents(event.work_id)) {
                for (const std::string& dependent_id : *dependents) {
                    if (dependent_id == event.work_id) continue;
                    const CommittedWork* dependent =
                        tx_shadow_work(tx, dependent_id);
                    if (dependent != nullptr
                            && dependent->state != WorkState::Released) {
                        still_depended = true;
                        break;
                    }
                }
            }
            if (still_depended) {
                tx.obligations_deleted.erase(event.work_id);
                tx.obligations_writes[event.work_id] = work->release_conditions;
            }
        }
        for (const std::string& dependency : work->dependencies) {
            const std::unordered_set<std::string>* current =
                view_dependents(dependency);
            if (current == nullptr) continue;
            std::unordered_set<std::string> next = *current;
            next.erase(event.work_id);
            if (next.empty()) {
                tx.dependents_writes.erase(dependency);
                tx.dependents_deleted.insert(dependency);
                // The retained release conditions die with the last
                // reference (work_ledger.py 1526-1533).  Python's overlay
                // pop(key, None) only marks deleted when the key is actually
                // present (overlay write or base); absent keys are a no-op.
                if (tx.obligations_writes.count(dependency) != 0
                        || retained_obligations_.count(dependency) != 0) {
                    tx.obligations_writes.erase(dependency);
                    tx.obligations_deleted.insert(dependency);
                }
            } else {
                tx.dependents_deleted.erase(dependency);
                tx.dependents_writes[dependency] = std::move(next);
            }
        }
        // The retired work object leaves the active set (bounded memory).
        tx.work_writes.erase(event.work_id);
        tx.work_deleted.insert(event.work_id);
        return;
    }
    // Python silently no-ops on kinds outside the six-event closed set (the
    // reader has already validated the kind); the native core fails closed.
    throw LedgerError("unknown ledger event kind '" + event.kind + "'");
}

// ------------------------------------------------------------------ queries --

Json WorkLedgerCore::pricing_json() const {
    Json rows = Json::array();
    std::uint64_t committed_not_issued = 0;
    for (const std::string& id : works_order_) {
        if (works_dead_.count(id)) continue;
        const auto it = works_.find(id);
        if (it == works_.end()) continue;
        const CommittedWork& work = it->second;
        if (work.state == WorkState::Committed
                || work.state == WorkState::Ready) {
            ++committed_not_issued;
        }
        if (!is_pricing_row_(work)) continue;
        Json row = Json::object();
        row["work_id"] = work.work_id();
        row["state"] = work_state_name(work.state);
        Json identity = Json::object();
        identity["request_id"] = work.identity.request_id;
        identity["stage"] = work.identity.stage;
        identity["generation"] = work.identity.generation;
        identity["rank"] = work.identity.rank;
        identity["node_id"] = work.identity.node_id;
        identity["operation"] = work.identity.operation;
        row["identity"] = std::move(identity);
        row["total"] = amount_json_(work.total);
        row["served"] = amount_json_(work.served);
        row["remaining"] = amount_json_(work.remaining);
        row["progress_source"] = work.progress_source;
        row["service_unit"] = work.service_unit;
        row["dependencies"] = work.dependencies;
        row["release_conditions"] = work.release_conditions;
        row["route"] = work.route ? Json(*work.route) : Json();
        row["weight"] = work.weight;
        row["leg"] = work.leg ? Json(*work.leg) : Json();
        row["committed_tick"] = optional_tick_json_(work.committed_tick);
        row["issued_tick"] = optional_tick_json_(work.issued_tick);
        row["completed_tick"] = optional_tick_json_(work.completed_tick);
        row["acked_tick"] = optional_tick_json_(work.acked_tick);
        row["released_tick"] = optional_tick_json_(work.released_tick);
        row["as_of_tick"] = optional_tick_json_(work.as_of_tick);
        rows.push_back(std::move(row));
    }
    Json out = Json::object();
    out["pricing_rows"] = std::move(rows);
    out["committed_not_issued"] = committed_not_issued;
    return out;
}

// work_ledger.conservation_report (1728-1774) 1:1.
Json WorkLedgerCore::conservation_json() const {
    Json violations = Json::array();
    std::uint64_t inferred_count = 0;
    std::uint64_t stale_count = 0;
    for (const std::string& id : works_order_) {
        if (works_dead_.count(id)) continue;
        const auto it = works_.find(id);
        if (it == works_.end()) continue;
        const CommittedWork& work = it->second;
        if (work.progress_source == "inferred") {
            ++inferred_count;
        } else if (work.progress_source == "stale") {
            ++stale_count;
        }
        for (int dim = 0; dim < 3; ++dim) {
            const std::optional<double> total = amount_dim(work.total, dim);
            if (!total) continue;
            const std::optional<double> served = amount_dim(work.served, dim);
            const std::optional<double> remaining =
                amount_dim(work.remaining, dim);
            const double served_v = served ? *served : 0.0;
            const double remaining_v =
                remaining ? *remaining : *total - served_v;
            if (std::fabs(served_v + remaining_v - *total)
                    > kAmountEpsilon
                        + kRelativeTolerance * std::fabs(*total)) {
                Json violation = Json::object();
                violation["work_id"] = work.work_id();
                violation["dimension"] = kDimensionNames[dim];
                violation["total"] = *total;
                violation["served"] = served_v;
                violation["remaining"] = remaining_v;
                violations.push_back(std::move(violation));
            }
        }
    }
    Json out = Json::object();
    out["violations"] = violations;  // copy: "conserved" reads the size
    out["conserved"] = violations.empty();
    out["inferred_progress_works"] = inferred_count;
    out["stale_amount_works"] = stale_count;
    out["active_works"] = works_.size();
    return out;
}

// work_ledger.export_state (1556-1718) 1:1.
Json WorkLedgerCore::works_export_json() const {
    std::map<std::int64_t, Json> nodes_by_rank;
    std::map<std::int64_t, Json> hbm_by_rank;
    std::map<std::int64_t, Json> pool_by_port;
    Json noc_flows = Json::array();
    std::map<std::int64_t, std::uint64_t> link_flow_counts;
    std::optional<std::int64_t> noc_as_of;

    const auto total_or_zero = [](const std::optional<double>& value) {
        return value ? Json(*value) : Json(0);
    };
    const auto raw_or_null = [](const std::optional<double>& value) {
        return value ? Json(*value) : Json();
    };

    for (const std::string& id : works_order_) {
        if (works_dead_.count(id)) continue;
        const auto it = works_.find(id);
        if (it == works_.end()) continue;
        const CommittedWork& work = it->second;
        const WorkIdentity& identity = work.identity;
        const bool issued =
            work.state == WorkState::Issued || work.state == WorkState::InService;
        const bool finished = work.state == WorkState::Completed
            || work.state == WorkState::Acknowledged
            || work.state == WorkState::Released;
        if (!work.leg) {
            Json node = Json::object();
            node["work_id"] = work.work_id();
            node["request_id"] = identity.request_id;
            node["stage"] = identity.stage;
            node["generation"] = identity.generation;
            node["rank"] = identity.rank;
            node["node_id"] = identity.node_id;
            node["kind"] = !identity.operation.empty() ? identity.operation
                                                       : std::string("invalid");
            node["issued"] = issued;
            node["finished"] = finished;
            node["num_ops"] = total_or_zero(work.total.ops);
            node["bytes"] = total_or_zero(work.total.bytes);
            node["runtime_ns"] = total_or_zero(work.total.latency_ns);
            node["dependencies"] = work.dependencies;
            node["progress_source"] = work.progress_source;
            const std::int64_t rank = static_cast<std::int64_t>(identity.rank);
            const auto slot = nodes_by_rank.find(rank);
            if (slot == nodes_by_rank.end()) {
                nodes_by_rank.emplace(
                    rank, Json::array({std::move(node)}));
            } else {
                slot->second.push_back(std::move(node));
            }
            continue;
        }
        const std::string& leg = *work.leg;
        if (leg.rfind("hbm:", 0) == 0) {
            const std::int64_t rank = static_cast<std::int64_t>(identity.rank);
            Json* block = nullptr;
            const auto found = hbm_by_rank.find(rank);
            if (found == hbm_by_rank.end()) {
                Json fresh = Json::object();
                fresh["rank"] = rank;
                fresh["as_of_tick"] = Json();
                fresh["jobs"] = Json::array();
                block = &hbm_by_rank.emplace(rank, std::move(fresh))
                             .first->second;
            } else {
                block = &found->second;
            }
            Json job = Json::object();
            job["work_id"] = work.work_id();
            job["node_id"] = identity.node_id;
            job["kind"] = leg.substr(leg.find(':') + 1);
            job["request_id"] = identity.request_id;
            job["stage"] = identity.stage;
            job["generation"] = identity.generation;
            job["num_ops"] = total_or_zero(work.total.ops);
            job["bytes"] = total_or_zero(work.total.bytes);
            job["total_latency_ns"] = total_or_zero(work.total.latency_ns);
            job["remaining_ops"] = raw_or_null(work.remaining.ops);
            job["remaining_bytes"] = raw_or_null(work.remaining.bytes);
            job["remaining_latency_ns"] =
                raw_or_null(work.remaining.latency_ns);
            job["state"] = work_state_name(work.state);
            job["progress_source"] = work.progress_source;
            block->at("jobs").push_back(std::move(job));
            if (work.as_of_tick) {
                const std::int64_t as_of =
                    static_cast<std::int64_t>(*work.as_of_tick);
                const Json& current = block->at("as_of_tick");
                if (current.is_null()
                        || as_of > current.get<std::int64_t>()) {
                    block->at("as_of_tick") = as_of;
                }
            }
            continue;
        }
        if (leg.rfind("pool:", 0) == 0) {
            // leg encodes port:issue_seq; malformed ids fail closed.
            std::vector<std::string> parts;
            std::size_t begin = 0;
            while (true) {
                const std::size_t colon = leg.find(':', begin);
                if (colon == std::string::npos) {
                    parts.push_back(leg.substr(begin));
                    break;
                }
                parts.push_back(leg.substr(begin, colon - begin));
                begin = colon + 1;
            }
            std::int64_t port = 0;
            std::int64_t issue_seq = 0;
            if (parts.size() < 2 || !parse_int_segment_(parts[1], &port)
                    || (parts.size() > 2
                        && !parse_int_segment_(parts[2], &issue_seq))) {
                throw LedgerError("pool leg id '" + leg
                                  + "' must encode port:issue_seq");
            }
            Json* block = nullptr;
            const auto found = pool_by_port.find(port);
            if (found == pool_by_port.end()) {
                Json fresh = Json::object();
                fresh["port_index"] = port;
                fresh["as_of_ns"] = Json();
                fresh["jobs"] = Json::array();
                block =
                    &pool_by_port.emplace(port, std::move(fresh)).first->second;
            } else {
                block = &found->second;
            }
            Json job = Json::object();
            job["work_id"] = work.work_id();
            job["issue_seq"] = issue_seq;
            job["sys_id"] = identity.rank;
            job["node_id"] = identity.node_id;
            job["request_id"] = identity.request_id;
            job["stage"] = identity.stage;
            job["generation"] = identity.generation;
            job["bytes"] = total_or_zero(work.total.bytes);
            job["remaining_bytes"] = raw_or_null(work.remaining.bytes);
            job["state"] = work.state == WorkState::InService ? "Streaming"
                                                              : "LatencyWaiting";
            job["progress_source"] = work.progress_source;
            block->at("jobs").push_back(std::move(job));
            if (work.as_of_tick) {
                const std::int64_t as_of =
                    static_cast<std::int64_t>(*work.as_of_tick);
                const Json& current = block->at("as_of_ns");
                if (current.is_null()
                        || as_of > current.get<std::int64_t>()) {
                    block->at("as_of_ns") = as_of;
                }
            }
            continue;
        }
        if (leg.rfind("noc:", 0) == 0) {
            Json flow = Json::object();
            flow["work_id"] = work.work_id();
            flow["flow_id"] = identity.node_id;
            flow["request_id"] = identity.request_id;
            flow["stage"] = identity.stage;
            flow["generation"] = identity.generation;
            flow["total_bytes"] = total_or_zero(work.total.bytes);
            flow["remaining_bytes"] = raw_or_null(work.remaining.bytes);
            Json route = Json::array();
            std::vector<std::int64_t> links;
            if (work.route) {
                for (const std::uint64_t link : *work.route) {
                    route.push_back(link);
                    links.push_back(static_cast<std::int64_t>(link));
                }
            }
            flow["route"] = std::move(route);
            flow["state"] = work_state_name(work.state);
            flow["progress_source"] = work.progress_source;
            noc_flows.push_back(std::move(flow));
            // links[] aggregates distinct route members per active flow.
            std::sort(links.begin(), links.end());
            links.erase(std::unique(links.begin(), links.end()), links.end());
            for (const std::int64_t link : links) {
                ++link_flow_counts[link];
            }
            if (work.as_of_tick) {
                const std::int64_t as_of =
                    static_cast<std::int64_t>(*work.as_of_tick);
                if (!noc_as_of || as_of > *noc_as_of) noc_as_of = as_of;
            }
            continue;
        }
        // Unknown leg prefixes are skipped (no invented fourth family).
    }

    Json out = Json::object();
    Json nodes = Json::array();
    for (const auto& entry : nodes_by_rank) {
        Json group = Json::object();
        group["rank"] = entry.first;
        group["nodes"] = entry.second;
        nodes.push_back(std::move(group));
    }
    out["nodes"] = std::move(nodes);
    Json hbm = Json::array();
    for (const auto& entry : hbm_by_rank) hbm.push_back(entry.second);
    out["hbm"] = std::move(hbm);
    Json pool = Json::array();
    for (const auto& entry : pool_by_port) pool.push_back(entry.second);
    out["pool"] = std::move(pool);
    Json noc = Json::object();
    noc["as_of_tick"] = noc_as_of ? Json(*noc_as_of) : Json();
    noc["flows"] = std::move(noc_flows);
    Json links_json = Json::array();
    for (const auto& entry : link_flow_counts) {
        Json link = Json::object();
        link["link_id"] = entry.first;
        link["active_flow_count"] = entry.second;
        links_json.push_back(std::move(link));
    }
    noc["links"] = std::move(links_json);
    out["noc"] = std::move(noc);
    out["slots"] = Json::array();
    return out;
}

}  // namespace jointledger
