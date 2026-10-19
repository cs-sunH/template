// LedgerTypes.cc -- shared value types and strict wire readers.
#include "joint_ledger/LedgerTypes.hh"

#include <cmath>

namespace jointledger {

// ---------------------------------------------------------------- states --

const char* work_state_name(WorkState state) {
    switch (state) {
        case WorkState::Committed: return "committed";
        case WorkState::Ready: return "ready";
        case WorkState::Issued: return "issued";
        case WorkState::InService: return "in_service";
        case WorkState::Completed: return "completed";
        case WorkState::Acknowledged: return "acknowledged";
        case WorkState::Released: return "released";
    }
    return "?";
}

WorkState work_state_from_name(const std::string& name) {
    if (name == "committed") return WorkState::Committed;
    if (name == "ready") return WorkState::Ready;
    if (name == "issued") return WorkState::Issued;
    if (name == "in_service") return WorkState::InService;
    if (name == "completed") return WorkState::Completed;
    if (name == "acknowledged") return WorkState::Acknowledged;
    if (name == "released") return WorkState::Released;
    throw LedgerError("unknown work state '" + name + "'");
}

// -------------------------------------------------------------- identity --

std::string canonical_work_id(const std::string& request_id,
                              const std::string& stage,
                              std::uint64_t generation,
                              std::uint64_t rank, std::uint64_t node_id) {
    // "{}#{}:{}:r{}n{}" -- identical formatting to work_ledger.py.
    return request_id + "#" + stage + ":" + std::to_string(generation) + ":r"
        + std::to_string(rank) + "n" + std::to_string(node_id);
}

std::string leg_work_id(const std::string& work_id, const std::string& leg) {
    return work_id + "@" + leg;
}

std::string CommittedWork::work_id() const {
    const std::string base = canonical_work_id(
        identity.request_id, identity.stage, identity.generation,
        identity.rank, identity.node_id);
    if (!leg) return base;
    return leg_work_id(base, *leg);
}

// Python _TRANSITIONS table (work_ledger.py).
static bool transition_allowed(WorkState from, WorkState to) {
    switch (from) {
        case WorkState::Committed:
            return to == WorkState::Ready || to == WorkState::Issued
                || to == WorkState::InService || to == WorkState::Completed;
        case WorkState::Ready:
            return to == WorkState::Issued || to == WorkState::InService
                || to == WorkState::Completed;
        case WorkState::Issued:
            return to == WorkState::InService || to == WorkState::Completed;
        case WorkState::InService:
            return to == WorkState::Completed;
        case WorkState::Completed:
            return to == WorkState::Acknowledged || to == WorkState::Released;
        case WorkState::Acknowledged:
            return to == WorkState::Released;
        case WorkState::Released:
            return false;
    }
    return false;
}

void CommittedWork::transition(WorkState target, std::uint64_t tick) {
    if (state == target) return;
    if (!transition_allowed(state, target)) {
        throw LedgerError(std::string("illegal transition '")
            + work_state_name(state) + "' -> '" + work_state_name(target)
            + "' for " + work_id());
    }
    state = target;
    if (target == WorkState::Issued && !issued_tick) issued_tick = tick;
    if (target == WorkState::Completed) completed_tick = tick;
    if (target == WorkState::Acknowledged) acked_tick = tick;
    if (target == WorkState::Released) released_tick = tick;
}

static void validate_amount(const WorkAmount& amount, const char* what) {
    for (const std::optional<double>* dim :
         {&amount.ops, &amount.bytes, &amount.latency_ns}) {
        if (!*dim) continue;
        const double value = **dim;
        if (!std::isfinite(value) || value < -1e-6) {
            throw LedgerError(std::string("work amount '") + what
                + "' must be finite and non-negative");
        }
    }
}

static std::optional<double> opt_add(const std::optional<double>& a,
                                     const std::optional<double>& b) {
    if (!a) return b;
    if (!b) return a;
    return *a + *b;
}

static std::optional<double> opt_sub(const std::optional<double>& a,
                                     const std::optional<double>& b) {
    if (!a || !b) return a;
    return *a - *b;
}

void CommittedWork::apply_progress(const WorkAmount& delta,
                                   std::uint64_t tick,
                                   const std::string& source) {
    if (state == WorkState::Completed || state == WorkState::Acknowledged
        || state == WorkState::Released) {
        throw LedgerError("progress event on finished work " + work_id());
    }
    const std::optional<double>* delta_dims[3] = {
        &delta.ops, &delta.bytes, &delta.latency_ns};
    const std::optional<double>* remaining_dims[3] = {
        &remaining.ops, &remaining.bytes, &remaining.latency_ns};
    for (int i = 0; i < 3; ++i) {
        if (*delta_dims[i]
                && !*remaining_dims[i]) {
            throw LedgerError("progress event " + work_id()
                + " serves an unavailable dimension");
        }
    }
    WorkAmount next_served{opt_add(served.ops, delta.ops),
                           opt_add(served.bytes, delta.bytes),
                           opt_add(served.latency_ns, delta.latency_ns)};
    validate_amount(next_served, "served");
    WorkAmount next_remaining{opt_sub(remaining.ops, delta.ops),
                              opt_sub(remaining.bytes, delta.bytes),
                              opt_sub(remaining.latency_ns, delta.latency_ns)};
    validate_amount(next_remaining, "remaining");
    served = next_served;
    remaining = next_remaining;
    if (progress_source != "authoritative") progress_source = source;
    if (state == WorkState::Committed || state == WorkState::Ready
        || state == WorkState::Issued) {
        state = WorkState::InService;
    }
    if (!issued_tick) issued_tick = tick;
}

void CommittedWork::finish(std::uint64_t tick) {
    if (state != WorkState::Completed && state != WorkState::Acknowledged
        && state != WorkState::Released) {
        transition(WorkState::Completed, tick);
    }
    remaining = WorkAmount{};
}

// --------------------------------------------------------------- readers --

// nlohmann is_boolean/is_number_integer/is_number_unsigned distinctions map
// exactly onto the Python bool/int split.

std::uint64_t read_uint(const Json& value, const std::string& path,
                        std::uint64_t maximum) {
    if (value.is_boolean()
        || !(value.is_number_unsigned() || value.is_number_integer())) {
        throw LedgerError(path + " must be uint in [0," + std::to_string(maximum)
                          + "] (bool rejected)");
    }
    if (value.is_number_unsigned()) {
        const std::uint64_t raw = value.get<std::uint64_t>();
        if (raw > maximum) {
            throw LedgerError(path + " must be uint in [0,"
                              + std::to_string(maximum) + "]");
        }
        return raw;
    }
    const std::int64_t raw = value.get<std::int64_t>();
    if (raw < 0 || static_cast<std::uint64_t>(raw) > maximum) {
        throw LedgerError(path + " must be uint in [0,"
                          + std::to_string(maximum) + "]");
    }
    return static_cast<std::uint64_t>(raw);
}

std::uint64_t read_rank(const Json& value, const std::string& path) {
    return read_uint(value, path, I32_MAX);
}

double read_finite_nonnegative(const Json& value, const std::string& path) {
    if (value.is_boolean()
        || !(value.is_number_integer() || value.is_number_float())) {
        throw LedgerError(
            path + " must be finite nonnegative int/float (bool rejected)");
    }
    const double number = value.get<double>();
    if (!std::isfinite(number) || number < 0) {
        throw LedgerError(
            path + " must be finite nonnegative int/float (bool rejected)");
    }
    return number;
}

std::string read_string(const Json& value, const std::string& path,
                        bool allow_empty) {
    if (!value.is_string()) {
        throw LedgerError(path + std::string(" must be a ")
                          + (allow_empty ? "string" : "non-empty string"));
    }
    const std::string text = value.get<std::string>();
    if (!allow_empty && text.empty()) {
        throw LedgerError(path + " must be a non-empty string");
    }
    return text;
}

bool read_bool(const Json& value, const std::string& path) {
    if (!value.is_boolean()) {
        throw LedgerError(path + " must be bool");
    }
    return value.get<bool>();
}

std::optional<std::uint64_t> read_nullable_uint(const Json& value,
                                                const std::string& path) {
    if (value.is_null()) return std::nullopt;
    return read_uint(value, path);
}

WorkAmount read_amount(const Json& value, const std::string& path) {
    if (!value.is_object()) {
        throw LedgerError(path + " must be an object");
    }
    static const char* kDims[3] = {"ops", "bytes", "latency_ns"};
    WorkAmount amount;
    for (int i = 0; i < 3; ++i) {
        const auto member = value.find(kDims[i]);
        if (member == value.end()) {
            throw LedgerError(path + " keys invalid (missing="
                              + std::string(kDims[i]) + ")");
        }
        if (member->is_null()) continue;
        // _finite_nonnegative keeps int/float distinction only for wire
        // comparisons; the stored value is float(number) like _amount_number.
        amount.*(i == 0 ? &WorkAmount::ops
                        : i == 1 ? &WorkAmount::bytes
                                 : &WorkAmount::latency_ns)
            = read_finite_nonnegative(*member, path + "." + kDims[i]);
    }
    if (value.size() != 3) {
        throw LedgerError(path + " keys invalid (unknown keys present)");
    }
    return amount;
}

// Python _typed_equal: type-sensitive structural equality.
bool typed_equal(const Json& left, const Json& right) {
    if (left.is_boolean() || right.is_boolean()) {
        return left.is_boolean() && right.is_boolean()
            && left.get<bool>() == right.get<bool>();
    }
    const bool left_int = left.is_number_integer() || left.is_number_unsigned();
    const bool right_int =
        right.is_number_integer() || right.is_number_unsigned();
    if (left_int || right_int) {
        if (!left_int || !right_int) return false;
        // Both Python ints; compare exactly across the signed/unsigned split.
        if (left.is_number_unsigned() && right.is_number_unsigned()) {
            return left.get<std::uint64_t>() == right.get<std::uint64_t>();
        }
        if (left.is_number_unsigned()) {
            const std::uint64_t l = left.get<std::uint64_t>();
            const std::int64_t r = right.get<std::int64_t>();
            return r >= 0 && l == static_cast<std::uint64_t>(r);
        }
        if (right.is_number_unsigned()) {
            const std::int64_t l = left.get<std::int64_t>();
            const std::uint64_t r = right.get<std::uint64_t>();
            return l >= 0 && static_cast<std::uint64_t>(l) == r;
        }
        return left.get<std::int64_t>() == right.get<std::int64_t>();
    }
    if (left.is_number_float() || right.is_number_float()) {
        return left.is_number_float() && right.is_number_float()
            && left.get<double>() == right.get<double>();
    }
    if (left.is_string() || right.is_string()) {
        return left.is_string() && right.is_string()
            && left.get<std::string>() == right.get<std::string>();
    }
    if (left.is_null() || right.is_null()) {
        return left.is_null() && right.is_null();
    }
    if (left.is_array() && right.is_array()) {
        if (left.size() != right.size()) return false;
        return std::equal(left.begin(), left.end(), right.begin(),
                          [](const Json& a, const Json& b) {
                              return typed_equal(a, b);
                          });
    }
    if (left.is_object() && right.is_object()) {
        if (left.size() != right.size()) return false;
        for (auto member = left.begin(); member != left.end(); ++member) {
            const auto other = right.find(member.key());
            if (other == right.end() || !typed_equal(member.value(), *other)) {
                return false;
            }
        }
        return true;
    }
    return false;
}

const Json& read_mapping(const Json& value, const std::string& path) {
    if (!value.is_object()) {
        throw LedgerError(path + " must be an object");
    }
    return value;
}

void require_keys(const Json& object, const std::string& path,
                  const std::vector<const char*>& allowed,
                  const std::vector<const char*>& required) {
    if (!object.is_object()) {
        throw LedgerError(path + " must be an object");
    }
    for (const char* key : required) {
        if (object.find(key) == object.end()) {
            throw LedgerError(path + " keys invalid (missing=" + key + ")");
        }
    }
    for (auto member = object.begin(); member != object.end(); ++member) {
        bool known = false;
        for (const char* key : allowed) {
            if (member.key() == key) {
                known = true;
                break;
            }
        }
        if (!known) {
            throw LedgerError(path + " keys invalid (unknown=" + member.key()
                              + ")");
        }
    }
}

}  // namespace jointledger
