// RowParsers.cc -- v4 envelope row validators, descriptor indexes and the
// Python-compatible canonical serializer (joint v4 acceleration round).
//
// 1:1 port of the row validators in sh_test_mesh/workload/llama2_7b_inference/
// joint/pipeline_ledger.py: _parse_one_pipeline_run/_resolve_selector,
// _parse_v4_slot_patches, _parse_row_patches, _parse_slot_rows,
// _parse_credit_rows, _parse_obligations, _parse_cursor_patch(es)/
// _parse_cursor_patch_run, _parse_resources, _build_descriptor_selector_index,
// _build_descriptor_dependency_index and _canonical_envelope_bytes (Python
// json.dumps(sort_keys=True, separators=(",",":"), ensure_ascii=False,
// allow_nan=False) byte-for-byte, including Python repr(float)).  The readers
// in LedgerTypes.cc (read_uint/read_rank/read_string/read_bool/
// read_nullable_uint/read_finite_nonnegative/read_amount/typed_equal/
// read_mapping/require_keys) are the ports of the same-named Python helpers
// and are used directly.
//
// The section entry points in namespace rowport below have external linkage
// so EnvelopeReader.cc (SAX router) and the tests can call them; the
// declarations are mirrored verbatim in EnvelopeReader.cc.
#include "joint_ledger/PipelineRows.hh"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace jointledger {
namespace rowport {

// ------------------------------------------------------------------ views --
// Merged view of this delivery's declarations over the retained registry
// (Python ChainMap(declared, retained)).  Defined identically in
// EnvelopeReader.cc (ODR-safe: same tokens, inline members only).

struct DescriptorView {
    const std::unordered_map<std::string, DescriptorPtr>* declared = nullptr;
    const std::unordered_map<std::string, DescriptorPtr>* retained = nullptr;
    const DescriptorPtr* find(const std::string& segment) const {
        if (declared != nullptr) {
            auto it = declared->find(segment);
            if (it != declared->end()) return &it->second;
        }
        if (retained != nullptr) {
            auto it = retained->find(segment);
            if (it != retained->end()) return &it->second;
        }
        return nullptr;
    }
};

struct SelectorView {
    const std::unordered_map<std::string, SelectorIndex>* declared = nullptr;
    const std::unordered_map<std::string, SelectorIndex>* retained = nullptr;
    const SelectorIndex* find(const std::string& segment) const {
        if (declared != nullptr) {
            auto it = declared->find(segment);
            if (it != declared->end()) return &it->second;
        }
        if (retained != nullptr) {
            auto it = retained->find(segment);
            if (it != retained->end()) return &it->second;
        }
        return nullptr;
    }
};

// Cross-row section cursors shared with EnvelopeReader.cc (declared there
// with the same tokens).
struct RunSeqCursor {
    bool has_previous = false;
    std::uint64_t previous_seq = 0;
};

struct CursorSegmentCursor {
    bool has_previous = false;
    std::string previous_segment;
};

using ResourceSortKey = std::tuple<std::string, std::uint64_t,
                                   std::uint64_t, std::string, std::string>;

namespace {

// ------------------------------------------------------------ key sets ----
// Exact key sets (pipeline_ledger.py module constants).

const std::vector<const char*>& run_keys() {
    static const std::vector<const char*> keys = {
        "first_logical_seq", "pattern_id", "repeat_count", "tile_first",
        "tile_stride", "rank_first", "rank_stride", "shard_first",
        "shard_stride", "transition_kind", "descriptor_segment_id",
        "endpoint_selector", "store_id_first", "store_id_stride",
        "legacy_context", "tick_first", "tick_stride", "resource_leg",
        "resource_quantity_prefix", "cursor_version_start",
        "cursor_version_end", "slot_patches", "credit_patches",
        "obligation_patches"};
    return keys;
}

const std::vector<const char*>& selector_keys() {
    static const std::vector<const char*> keys = {
        "group_ordinal", "tile_ordinal", "shard_ordinal", "endpoint"};
    return keys;
}

const std::vector<const char*>& slot_keys() {
    static const std::vector<const char*> keys = {
        "buffer_domain_id", "role", "rank", "shard_ordinal", "slot_index",
        "occupied", "owner", "consumed", "target_shards_to_consume",
        "consumed_target_shards"};
    return keys;
}

const std::vector<const char*>& owner_keys() {
    static const std::vector<const char*> keys = {
        "segment_id", "pipeline_generation", "stage_epoch",
        "staging_allocation_id", "group_ordinal", "tile_ordinal"};
    return keys;
}

const std::vector<const char*>& target_key_keys() {
    static const std::vector<const char*> keys = {
        "buffer_domain_id", "role", "rank", "shard_ordinal", "slot_index"};
    return keys;
}

const std::vector<const char*>& target_delta_keys() {
    static const std::vector<const char*> keys = {
        "form", "key", "before_slot_generation", "predecessor_logical_seq",
        "predecessor_consumed_count", "operations"};
    return keys;
}

const std::vector<const char*>& credit_keys() {
    static const std::vector<const char*> keys = {
        "buffer_domain_id", "source_rank", "shard_ordinal", "capacity",
        "available", "outstanding_ack_count"};
    return keys;
}

const std::vector<const char*>& obligation_keys() {
    static const std::vector<const char*> keys = {
        "obligation_id", "segment_id", "rank", "first_store_id",
        "remaining_endpoint_count", "state"};
    return keys;
}

const std::vector<const char*>& cursor_run_keys() {
    static const std::vector<const char*> keys = {
        "rank", "group_ordinal", "first_tile", "tile_count",
        "completed_endpoint_count", "cancelled_endpoint_count",
        "remaining_endpoint_count", "in_flight", "next_store_id",
        "tile_offset", "shard_index", "leg_index",
        "initial_parent_store_id", "completed_out_of_order",
        "active_endpoints"};
    return keys;
}

const std::vector<const char*>& active_endpoint_keys() {
    static const std::vector<const char*> keys = {
        "store_id", "tile_ordinal", "shard_ordinal", "endpoint",
        "transfer_id", "flow_id", "network_done", "hbm_done", "owner"};
    return keys;
}

const std::vector<const char*>& patch_keys() {
    static const std::vector<const char*> keys = {"before", "after"};
    return keys;
}

const std::vector<const char*>& resource_common_keys() {
    static const std::vector<const char*> keys = {
        "resource_type", "work_id", "request_id", "stage", "generation",
        "rank", "node_id", "operation", "leg", "state", "total", "remaining",
        "progress_source", "service_unit", "route", "as_of_tick",
        "issued_tick", "descriptor_segment_id", "endpoint_selector"};
    return keys;
}

const char* const kTransitions[] = {
    "descriptor_commit", "tile_issue", "data_ready", "endpoint_done",
    "fill_ack_progress", "fill_ack_done", "consume_start", "consume_done",
    "release_ack_progress", "release_ack_done", "credit_return",
    "service_done", "request_close", "settle_done", "descriptor_retire"};

const char* const kResourceLegs[] = {
    "endpoint", "body_marker", "noc", "hbm:COMM_READ", "hbm:COMM_WRITE"};

bool in_strings(const Json& value, const char* const* list, std::size_t count) {
    if (!value.is_string()) return false;
    const std::string& text = value.get_ref<const std::string&>();
    for (std::size_t i = 0; i < count; ++i) {
        if (text == list[i]) return true;
    }
    return false;
}

void require_exact_keys(const Json& object, const std::string& path,
                        const std::vector<const char*>& keys) {
    require_keys(object, path, keys, keys);
}

bool json_is_int(const Json& v) {
    return v.is_number_integer() || v.is_number_unsigned();
}

std::int64_t json_int_value(const Json& v) {
    return v.get<std::int64_t>();
}

// Python numeric comparison for sort keys / ordering checks: every component
// of the ported sort keys is a validated string or number, so mixed kinds
// cannot occur (Python would raise TypeError there).
int json_scalar_cmp(const Json& a, const Json& b) {
    if (a.is_string() && b.is_string()) {
        const std::string& x = a.get_ref<const std::string&>();
        const std::string& y = b.get_ref<const std::string&>();
        return x < y ? -1 : (x == y ? 0 : 1);
    }
    if (json_is_int(a) && json_is_int(b)) {
        const __int128 x = a.is_number_unsigned()
            ? static_cast<__int128>(a.get<std::uint64_t>())
            : static_cast<__int128>(a.get<std::int64_t>());
        const __int128 y = b.is_number_unsigned()
            ? static_cast<__int128>(b.get<std::uint64_t>())
            : static_cast<__int128>(b.get<std::int64_t>());
        return x < y ? -1 : (x == y ? 0 : 1);
    }
    const double x = a.get<double>();
    const double y = b.get<double>();
    return x < y ? -1 : (x == y ? 0 : 1);
}

// Python tuple <= over Json arrays (used for sorted/duplicate-free checks).
int json_tuple_cmp(const Json& a, const Json& b) {
    const std::size_t common = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < common; ++i) {
        const int cmp = json_scalar_cmp(a[i], b[i]);
        if (cmp != 0) return cmp;
    }
    if (a.size() < b.size()) return -1;
    if (a.size() > b.size()) return 1;
    return 0;
}

std::string path_index(const std::string& path, std::size_t index) {
    return path + "[" + std::to_string(index) + "]";
}

// Python str.isdigit for the port (ASCII digits only; the Python original
// also accepts exotic unicode digits whose int() then raises -- the port
// fails closed on those instead of crashing).
bool ascii_digits(const std::string& text) {
    if (text.empty()) return false;
    for (unsigned char c : text) {
        if (c < '0' || c > '9') return false;
    }
    return true;
}

// Decimal parse that never throws: false = value exceeds uint64 (Python
// int() is unbounded; callers treat overflow per the Python comparison).
bool parse_decimal_u64(const std::string& text, std::uint64_t& out) {
    if (!ascii_digits(text)) return false;
    std::size_t i = 0;
    while (i < text.size() && text[i] == '0') ++i;
    if (text.size() - i > 20) return false;
    const std::string digits = text.substr(i);
    try {
        out = digits.empty() ? 0 : std::stoull(digits);
    } catch (...) {
        return false;
    }
    return true;
}

// ------------------------------------------------------------ sha256 ------

struct Sha256 {
    std::uint32_t h[8];
    std::uint64_t bits = 0;
    unsigned char buffer[64];
    std::size_t buffered = 0;

    Sha256() {
        static const std::uint32_t init[8] = {0x6a09e667u, 0xbb67ae85u,
            0x3c6ef372u, 0xa54ff53au, 0x510e527fu, 0x9b05688cu, 0x1f83d9abu,
            0x5be0cd19u};
        std::memcpy(h, init, sizeof h);
    }

    static std::uint32_t ror(std::uint32_t x, int n) {
        return (x >> n) | (x << (32 - n));
    }

    void compress(const unsigned char* block) {
        static const std::uint32_t k[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
            0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
            0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
            0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
            0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
            0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
            0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
            0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
            0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
            0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
            0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
            0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<std::uint32_t>(block[i * 4]) << 24)
                | (static_cast<std::uint32_t>(block[i * 4 + 1]) << 16)
                | (static_cast<std::uint32_t>(block[i * 4 + 2]) << 8)
                | static_cast<std::uint32_t>(block[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18)
                ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19)
                ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        std::uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t s1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25);
            const std::uint32_t ch = (e & f) ^ (~e & g);
            const std::uint32_t t1 = hh + s1 + ch + k[i] + w[i];
            const std::uint32_t s0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22);
            const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t t2 = s0 + maj;
            hh = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    void update(const unsigned char* data, std::size_t length) {
        bits += static_cast<std::uint64_t>(length) * 8u;
        while (length > 0) {
            const std::size_t take = std::min<std::size_t>(
                length, 64 - buffered);
            std::memcpy(buffer + buffered, data, take);
            buffered += take;
            data += take;
            length -= take;
            if (buffered == 64) {
                compress(buffer);
                buffered = 0;
            }
        }
    }

    std::string hex_digest() {
        const std::uint64_t total_bits = bits;
        unsigned char pad = 0x80;
        update(&pad, 1);
        unsigned char zero = 0;
        while (buffered != 56) update(&zero, 1);
        unsigned char tail[8];
        for (int i = 0; i < 8; ++i) {
            tail[i] = static_cast<unsigned char>(total_bits >> (56 - 8 * i));
        }
        // The trailing length bytes must not re-enter the bit counter; the
        // buffer is exactly 56 bytes full here, so append and compress.
        std::memcpy(buffer + 56, tail, 8);
        compress(buffer);
        static const char* digits = "0123456789abcdef";
        std::string out;
        out.reserve(64);
        for (int i = 0; i < 8; ++i) {
            for (int shift = 24; shift >= 0; shift -= 8) {
                const unsigned char byte =
                    static_cast<unsigned char>(h[i] >> shift);
                out.push_back(digits[byte >> 4]);
                out.push_back(digits[byte & 0xF]);
            }
        }
        return out;
    }
};

}  // namespace

std::string sha256_hex(const std::string& message) {
    Sha256 state;
    state.update(reinterpret_cast<const unsigned char*>(message.data()),
                 message.size());
    return state.hex_digest();
}

// -------------------------------------------------- Python == semantics ---

namespace {

bool python_num_equal(const Json& a, const Json& b) {
    if (a.is_number_float() || b.is_number_float()) {
        return a.get<double>() == b.get<double>();
    }
    return typed_equal(a, b);  // both Python ints: exact compare
}

}  // namespace

bool python_equal(const Json& left, const Json& right) {
    const bool left_bool = left.is_boolean();
    const bool right_bool = right.is_boolean();
    if (left_bool || right_bool) {
        // Python: True == 1, False == 0 (bool is an int subclass).
        if (left_bool && right_bool) {
            return left.get<bool>() == right.get<bool>();
        }
        if (left_bool) {
            return json_is_int(right) || right.is_number_float()
                ? python_num_equal(Json(left.get<bool>() ? 1 : 0), right)
                : false;
        }
        return json_is_int(left) || left.is_number_float()
            ? python_num_equal(left, Json(right.get<bool>() ? 1 : 0))
            : false;
    }
    if (left.is_number() && right.is_number()) {
        return python_num_equal(left, right);
    }
    if (left.is_string() || right.is_string()) {
        return left.is_string() && right.is_string()
            && left.get_ref<const std::string&>()
                == right.get_ref<const std::string&>();
    }
    if (left.is_null() || right.is_null()) {
        return left.is_null() && right.is_null();
    }
    if (left.is_array() && right.is_array()) {
        if (left.size() != right.size()) return false;
        for (std::size_t i = 0; i < left.size(); ++i) {
            if (!python_equal(left[i], right[i])) return false;
        }
        return true;
    }
    if (left.is_object() && right.is_object()) {
        if (left.size() != right.size()) return false;
        for (auto member = left.begin(); member != left.end(); ++member) {
            const auto other = right.find(member.key());
            if (other == right.end()
                    || !python_equal(member.value(), *other)) {
                return false;
            }
        }
        return true;
    }
    return false;
}

// ---------------------------------------------------- canonical bytes -----

namespace {

// Python json.dumps with ensure_ascii=False: only '"' and '\\' plus control
// characters are escaped; \b\f\n\r\t use the short escapes, other control
// characters \u00xx (lowercase hex); everything else (incl. non-ASCII) is
// emitted as raw UTF-8.
void append_canonical_string(std::string& out, const std::string& text) {
    out.push_back('"');
    for (unsigned char c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buffer[8];
                    std::snprintf(buffer, sizeof buffer, "\\u%04x",
                                  static_cast<unsigned>(c));
                    out += buffer;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    out.push_back('"');
}

// Python repr(float): shortest round-trip digits reformatted with Python's
// fixed/scientific thresholds (scientific iff exponent < -4 or >= 16),
// ".0" for integral fixed values and at least two exponent digits.
std::string python_float_repr(double value) {
    if (std::isnan(value) || std::isinf(value)) {
        throw LedgerError(
            "envelope cannot be represented as canonical JSON: "
            "Out of range float values are not JSON compliant");
    }
    const bool negative = std::signbit(value);
    const double magnitude = std::fabs(value);
    if (magnitude == 0.0) return negative ? "-0.0" : "0.0";
    char buffer[64];
    const auto result = std::to_chars(buffer, buffer + sizeof buffer,
                                      magnitude, std::chars_format::scientific);
    const std::string scientific(buffer, result.ptr);
    const std::size_t e_pos = scientific.find('e');
    std::string digits = scientific.substr(0, e_pos);
    const int exponent = std::atoi(scientific.c_str() + e_pos + 1);
    digits.erase(std::remove(digits.begin(), digits.end(), '.'),
                 digits.end());
    std::string body;
    if (exponent >= 16 || exponent < -4) {
        body.push_back(digits[0]);
        if (digits.size() > 1) {
            body.push_back('.');
            body += digits.substr(1);
        }
        char exponent_text[16];
        std::snprintf(exponent_text, sizeof exponent_text, "e%+03d", exponent);
        body += exponent_text;
    } else if (exponent >= static_cast<int>(digits.size()) - 1) {
        body = digits
            + std::string(
                exponent - (static_cast<int>(digits.size()) - 1), '0')
            + ".0";
    } else if (exponent >= 0) {
        body = digits.substr(0, exponent + 1) + "." + digits.substr(exponent + 1);
    } else {
        body = "0." + std::string(-exponent - 1, '0') + digits;
    }
    return negative ? "-" + body : body;
}

void canonical_dump(const Json& value, std::string& out) {
    switch (value.type()) {
        case Json::value_t::null:
            out += "null";
            break;
        case Json::value_t::boolean:
            out += value.get<bool>() ? "true" : "false";
            break;
        case Json::value_t::number_integer:
            out += std::to_string(value.get<std::int64_t>());
            break;
        case Json::value_t::number_unsigned:
            out += std::to_string(value.get<std::uint64_t>());
            break;
        case Json::value_t::number_float:
            out += python_float_repr(value.get<double>());
            break;
        case Json::value_t::string:
            append_canonical_string(out,
                                    value.get_ref<const std::string&>());
            break;
        case Json::value_t::array: {
            out.push_back('[');
            bool first = true;
            for (const Json& element : value) {
                if (!first) out.push_back(',');
                first = false;
                canonical_dump(element, out);
            }
            out.push_back(']');
            break;
        }
        case Json::value_t::object: {
            std::vector<const std::string*> keys;
            keys.reserve(value.size());
            for (auto member = value.begin(); member != value.end();
                 ++member) {
                keys.push_back(&member.key());
            }
            std::sort(keys.begin(), keys.end(),
                      [](const std::string* a, const std::string* b) {
                          return *a < *b;  // memcmp order == code point order
                      });
            out.push_back('{');
            bool first = true;
            for (const std::string* key : keys) {
                if (!first) out.push_back(',');
                first = false;
                append_canonical_string(out, *key);
                out.push_back(':');
                canonical_dump(value[*key], out);
            }
            out.push_back('}');
            break;
        }
        default:
            throw LedgerError(
                "envelope cannot be represented as canonical JSON: "
                "unsupported value type");
    }
}

}  // namespace

std::string canonical_json_dumps(const Json& value) {
    std::string out;
    out.reserve(256);
    canonical_dump(value, out);
    return out;
}

// ---------------------------------------------------- selector indexes ----

namespace {

// pipeline_ledger.py _descriptor_run_at / _descriptor_selector_at
// (bisect_right(starts, tile) - 1 then start <= tile < end).
std::optional<std::uint64_t> descriptor_run_at(const SelectorIndex& index,
                                               std::uint64_t group,
                                               std::uint64_t rank,
                                               std::uint64_t tile) {
    const auto found = index.run_intervals.find(std::make_tuple(group, rank));
    if (found == index.run_intervals.end()) return std::nullopt;
    const std::vector<std::uint64_t>& starts = found->second.first;
    const std::vector<std::array<std::uint64_t, 3>>& rows = found->second.second;
    const auto position = std::upper_bound(starts.begin(), starts.end(), tile);
    if (position == starts.begin()) return std::nullopt;
    const std::size_t index_at = static_cast<std::size_t>(
        position - starts.begin()) - 1;
    const auto& row = rows[index_at];
    if (row[0] <= tile && tile < row[1]) return row[2];
    return std::nullopt;
}

struct RunLegMatch {
    std::uint64_t run_index;
    std::uint64_t leg_index;
};

std::optional<RunLegMatch> descriptor_selector_at(const SelectorIndex& index,
                                                  std::uint64_t group,
                                                  std::uint64_t rank,
                                                  std::uint64_t tile,
                                                  std::uint64_t shard,
                                                  const std::string& endpoint) {
    const auto found = index.selector_intervals.find(
        std::make_tuple(group, rank, shard, endpoint));
    if (found == index.selector_intervals.end()) return std::nullopt;
    const std::vector<std::uint64_t>& starts = found->second.first;
    const std::vector<std::array<std::uint64_t, 4>>& rows =
        found->second.second;
    const auto position = std::upper_bound(starts.begin(), starts.end(), tile);
    if (position == starts.begin()) return std::nullopt;
    const std::size_t index_at = static_cast<std::size_t>(
        position - starts.begin()) - 1;
    const auto& row = rows[index_at];
    if (row[0] <= tile && tile < row[1]) {
        return RunLegMatch{row[2], row[3]};
    }
    return std::nullopt;
}

// Python ``((selector_indexes or {}).get(segment)
//          or _build_descriptor_selector_index(descriptor))``: per-row rebuild
// on lookup miss, exactly like the Python callers.
const SelectorIndex* resolve_selector_index(const SelectorView& view,
                                            const std::string& segment,
                                            const PipelineDescriptor& descriptor,
                                            SelectorIndex& storage) {
    if (const SelectorIndex* found = view.find(segment)) return found;
    storage = build_selector_index(descriptor);
    return &storage;
}

}  // namespace

}  // namespace rowport

// Header-exported index builders live directly in jointledger
// (PipelineRows.hh), one namespace level above rowport.
SelectorIndex build_selector_index(const PipelineDescriptor& descriptor) {
    SelectorIndex index;
    for (std::size_t i = 0; i < descriptor.groups.size(); ++i) {
        index.groups_by_ordinal.emplace(descriptor.groups[i].group_ordinal, i);
    }
    std::map<std::tuple<std::uint64_t, std::uint64_t>,
             std::vector<std::array<std::uint64_t, 3>>> run_rows;
    std::map<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t,
                        std::string>,
             std::vector<std::array<std::uint64_t, 4>>> selector_rows;
    std::map<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>,
             std::vector<std::array<std::uint64_t, 4>>> consume_rows;
    for (std::size_t run_index = 0; run_index < descriptor.endpoint_runs.size();
         ++run_index) {
        const PipelineEndpointRun& run = descriptor.endpoint_runs[run_index];
        const std::uint64_t start = run.first_tile;
        const std::uint64_t end = start + run.tile_count;
        run_rows[std::make_tuple(run.group_ordinal, run.rank)]
            .push_back({start, end, static_cast<std::uint64_t>(run_index)});
        for (std::uint64_t shard : run.shard_ordinals) {
            for (std::size_t leg_index = 0; leg_index < run.legs.size();
                 ++leg_index) {
                selector_rows[std::make_tuple(
                    run.group_ordinal, run.rank, shard,
                    run.legs[leg_index].endpoint)]
                    .push_back({start, end,
                                static_cast<std::uint64_t>(run_index),
                                static_cast<std::uint64_t>(leg_index)});
            }
        }
    }
    for (std::size_t completion_index = 0;
         completion_index < descriptor.consume_completion_runs.size();
         ++completion_index) {
        const PipelineConsumeCompletionRun& completion =
            descriptor.consume_completion_runs[completion_index];
        const std::uint64_t start = completion.first_tile;
        const std::uint64_t end = start + completion.tile_count;
        for (std::size_t shard_position = 0;
             shard_position < completion.shard_ordinals.size();
             ++shard_position) {
            consume_rows[std::make_tuple(
                completion.rank, completion.group_ordinal,
                completion.shard_ordinals[shard_position])]
                .push_back({start, end,
                            static_cast<std::uint64_t>(completion_index),
                            static_cast<std::uint64_t>(shard_position)});
        }
    }
    auto freeze3 = [&index](auto& rows) {
        for (auto& entry : rows) {
            std::sort(entry.second.begin(), entry.second.end());
            std::vector<std::uint64_t> starts;
            starts.reserve(entry.second.size());
            for (const auto& row : entry.second) starts.push_back(row[0]);
            index.run_intervals.emplace(entry.first,
                                        std::make_pair(std::move(starts),
                                                       std::move(entry.second)));
        }
    };
    auto freeze4 = [](auto& rows, auto& target) {
        for (auto& entry : rows) {
            std::sort(entry.second.begin(), entry.second.end());
            std::vector<std::uint64_t> starts;
            starts.reserve(entry.second.size());
            for (const auto& row : entry.second) starts.push_back(row[0]);
            target.emplace(entry.first,
                           std::make_pair(std::move(starts),
                                          std::move(entry.second)));
        }
    };
    freeze3(run_rows);
    freeze4(selector_rows, index.selector_intervals);
    freeze4(consume_rows, index.consume_intervals_by_target);
    return index;
}

DependencyIndex build_dependency_index(const PipelineDescriptor& descriptor) {
    DependencyIndex index;
    std::unordered_map<std::string, std::vector<std::array<std::uint64_t, 9>>>
        rows;
    std::unordered_map<std::uint64_t, std::vector<std::array<std::uint64_t, 9>>>
        rank_rows;
    for (std::size_t run_index = 0; run_index < descriptor.endpoint_runs.size();
         ++run_index) {
        const PipelineEndpointRun& run = descriptor.endpoint_runs[run_index];
        const PipelineLegacyContext& context = run.legacy_context;
        const std::string prefix = context.request_id + "#" + context.stage
            + ":" + std::to_string(context.generation) + ":r"
            + std::to_string(run.rank) + "n";
        const std::uint64_t stride = run.tile_stride;
        if (stride == 0) {
            throw LedgerError("descriptor endpoint run has zero tile stride");
        }
        const std::uint64_t ordinal_tile_stride =
            run.shard_ordinals.size() * run.legs.size();
        for (std::size_t shard_index = 0; shard_index < run.shard_ordinals.size();
             ++shard_index) {
            const std::uint64_t shard = run.shard_ordinals[shard_index];
            for (std::size_t leg_index = 0; leg_index < run.legs.size();
                 ++leg_index) {
                const std::uint64_t first =
                    descriptor.endpoint_id(run_index, run.first_tile, shard,
                                           leg_index) + 1;
                const std::uint64_t last =
                    first + (run.tile_count - 1) * stride;
                // Python row[2] is the segment string; the frozen
                // DependencyIndex rows are uint64 9-tuples and the index
                // itself is keyed by segment_id, so the segment column
                // carries a placeholder zero here.
                const std::array<std::uint64_t, 9> interval = {
                    first, last, 0, run.rank, stride, first % stride,
                    static_cast<std::uint64_t>(run_index),
                    shard_index * run.legs.size() + leg_index,
                    ordinal_tile_stride};
                rows[prefix].push_back(interval);
                rank_rows[run.rank].push_back(interval);
            }
        }
    }
    for (auto& entry : rows) {
        std::sort(entry.second.begin(), entry.second.end());
        index.intervals_by_prefix.emplace(entry.first,
                                          std::move(entry.second));
    }
    for (auto& entry : rank_rows) {
        std::sort(entry.second.begin(), entry.second.end());
        index.intervals_by_rank.emplace(entry.first, std::move(entry.second));
    }
    return index;
}

namespace rowport {

// ------------------------------------------------------------ resolve -----

namespace {

struct SelectorResolution {
    std::uint64_t json_id = 0;
    std::uint64_t run_index = 0;
    std::uint64_t leg_index = 0;
    std::uint64_t group = 0;
    std::uint64_t tile = 0;
    std::uint64_t shard = 0;
    std::string endpoint;
};

// pipeline_ledger.py _resolve_selector.  ``rank`` mirrors the production
// callers' typed rank; when absent the small-helper scan path runs (all
// selector_intervals keys matching group/shard/endpoint, exactly one hit).
SelectorResolution resolve_selector(const PipelineDescriptor& descriptor,
                                    const Json& selector,
                                    const SelectorIndex& indexes,
                                    const std::optional<std::uint64_t>& rank) {
    SelectorResolution out;
    out.group = read_uint(selector["group_ordinal"],
                          "endpoint_selector.group_ordinal");
    out.tile = read_uint(selector["tile_ordinal"],
                         "endpoint_selector.tile_ordinal");
    out.shard = read_uint(selector["shard_ordinal"],
                          "endpoint_selector.shard_ordinal");
    out.endpoint = read_string(selector["endpoint"],
                               "endpoint_selector.endpoint");
    std::optional<RunLegMatch> match;
    if (rank.has_value()) {
        match = descriptor_selector_at(indexes, out.group, *rank, out.tile,
                                       out.shard, out.endpoint);
    } else {
        std::vector<RunLegMatch> matches;
        for (const auto& entry : indexes.selector_intervals) {
            const auto& key = entry.first;
            if (std::get<0>(key) == out.group
                    && std::get<2>(key) == out.shard
                    && std::get<3>(key) == out.endpoint) {
                const std::optional<RunLegMatch> found =
                    descriptor_selector_at(indexes, out.group,
                                           std::get<1>(key), out.tile,
                                           out.shard, out.endpoint);
                if (found.has_value()) matches.push_back(*found);
            }
        }
        if (matches.size() == 1) match = matches[0];
    }
    if (!match.has_value()) {
        throw LedgerError(
            "endpoint selector must map to exactly one descriptor endpoint");
    }
    out.run_index = match->run_index;
    out.leg_index = match->leg_index;
    out.json_id = descriptor.endpoint_id(out.run_index, out.tile, out.shard,
                                         out.leg_index);
    return out;
}

// ------------------------------------------------------------- amounts ----

// Python _amount: exact {ops,bytes,latency_ns} key set with per-dimension
// finite-nonnegative checks; the raw mapping is kept (int/float distinction
// survives for the typed_equal amount comparisons downstream).
const Json& amount_mapping(const Json& value, const std::string& path) {
    const Json& row = read_mapping(value, path);
    static const std::vector<const char*> dims = {"ops", "bytes", "latency_ns"};
    require_exact_keys(row, path, dims);
    for (const char* dim : dims) {
        if (!row[dim].is_null()) {
            read_finite_nonnegative(row[dim], path + "." + dim);
        }
    }
    return row;
}

Json expected_endpoint_amount(const PipelineDescriptor& descriptor,
                              std::size_t run_index,
                              std::uint64_t ordinal,
                              const PipelineShardTile& shard_tile) {
    const PipelineEndpointRun& run = descriptor.endpoint_runs[run_index];
    const std::uint64_t per_tile = run.shard_ordinals.size()
        * run.legs.size();
    const std::uint64_t within = ordinal % per_tile;
    const std::size_t leg_index = static_cast<std::size_t>(
        within % run.legs.size());
    const std::string& leg = run.legs[leg_index].endpoint;
    Json out = Json::object();
    if (leg == "body_marker") {
        out["ops"] = static_cast<std::uint64_t>(1);
        out["bytes"] = static_cast<std::uint64_t>(1);
        out["latency_ns"] = nullptr;
        return out;
    }
    std::uint64_t amount = shard_tile.data_bytes;
    if (leg == "fill_ack_send" || leg == "fill_ack_recv") {
        amount = descriptor.protocol.fill_ack_bytes;
    } else if (leg == "release_ack_send" || leg == "release_ack_recv") {
        amount = descriptor.protocol.release_ack_bytes;
    }
    out["ops"] = nullptr;
    out["bytes"] = amount;
    out["latency_ns"] = nullptr;
    return out;
}

// --------------------------------------------------------- slot rows ------

Json parse_slot_row(const Json& raw, const std::string& p,
                    const DescriptorView& descriptors,
                    const SelectorView& selector_indexes) {
    const Json& row = read_mapping(raw, p);
    require_exact_keys(row, p, slot_keys());
    const std::string domain = read_string(row["buffer_domain_id"],
                                           p + ".buffer_domain_id");
    const Json& role = row["role"];
    const bool role_source = role.is_string()
        && role.get_ref<const std::string&>() == "source";
    const bool role_target = role.is_string()
        && role.get_ref<const std::string&>() == "target";
    if (!role_source && !role_target) {
        throw LedgerError(p + ".role must be source or target");
    }
    const std::uint64_t rank = read_rank(row["rank"], p + ".rank");
    const std::optional<std::uint64_t> shard = read_nullable_uint(
        row["shard_ordinal"], p + ".shard_ordinal");
    if (role_source != shard.has_value()) {
        throw LedgerError(p + ".shard_ordinal nullability disagrees with role");
    }
    const std::uint64_t slot_index = read_uint(row["slot_index"],
                                               p + ".slot_index", 1);
    const bool occupied = read_bool(row["occupied"], p + ".occupied");
    const bool consumed = read_bool(row["consumed"], p + ".consumed");
    const Json& targets_wire = row["target_shards_to_consume"];
    if (!targets_wire.is_array()) {
        throw LedgerError(p + ".target_shards_to_consume must be an array");
    }
    std::vector<std::uint64_t> targets;
    for (std::size_t i = 0; i < targets_wire.size(); ++i) {
        targets.push_back(read_uint(targets_wire[i],
                                    p + ".target_shards_to_consume[]"));
    }
    const Json& consumed_wire = row["consumed_target_shards"];
    if (!consumed_wire.is_array()) {
        throw LedgerError(p + ".consumed_target_shards must be an array");
    }
    std::vector<std::uint64_t> consumed_targets;
    for (std::size_t i = 0; i < consumed_wire.size(); ++i) {
        consumed_targets.push_back(read_uint(
            consumed_wire[i], p + ".consumed_target_shards[]"));
    }
    const auto sorted_unique = [](const std::vector<std::uint64_t>& values) {
        for (std::size_t i = 1; i < values.size(); ++i) {
            if (values[i - 1] >= values[i]) return false;
        }
        return true;
    };
    if (!sorted_unique(targets) || !sorted_unique(consumed_targets)) {
        throw LedgerError(p + " target shard arrays must be sorted unique");
    }
    for (const std::uint64_t value : consumed_targets) {
        if (!std::binary_search(targets.begin(), targets.end(), value)) {
            throw LedgerError(p + ".consumed_target_shards is not a subset");
        }
    }
    const Json& owner_wire = row["owner"];
    if (occupied == owner_wire.is_null()) {
        throw LedgerError(p + ".owner must be nonnull iff occupied");
    }
    if (!owner_wire.is_null()) {
        const Json& owner = read_mapping(owner_wire, p + ".owner");
        require_exact_keys(owner, p + ".owner", owner_keys());
        const std::string segment = read_string(owner["segment_id"],
                                                p + ".owner.segment_id");
        const DescriptorPtr* found = descriptors.find(segment);
        if (found == nullptr) {
            throw LedgerError(p + ".owner references unknown segment '"
                              + segment + "'");
        }
        const PipelineDescriptor& descriptor = **found;
        const PipelineIdentity& ident = descriptor.identity;
        const std::uint64_t pipeline_generation = read_uint(
            owner["pipeline_generation"], p + ".owner.pipeline_generation");
        const std::uint64_t stage_epoch = read_uint(owner["stage_epoch"],
                                                    p + ".owner.stage_epoch");
        const std::uint64_t allocation = read_uint(
            owner["staging_allocation_id"],
            p + ".owner.staging_allocation_id");
        if (domain != ident.buffer_domain_id
                || pipeline_generation != ident.generation
                || stage_epoch != ident.stage_epoch
                || allocation != ident.staging_allocation_id) {
            throw LedgerError(
                p + ".owner disagrees with immutable descriptor identity");
        }
        const std::uint64_t group_ordinal = read_uint(
            owner["group_ordinal"], p + ".owner.group_ordinal");
        const std::uint64_t tile_ordinal = read_uint(owner["tile_ordinal"],
                                                     p + ".owner.tile_ordinal");
        SelectorIndex storage;
        const SelectorIndex* selector_index = resolve_selector_index(
            selector_indexes, segment, descriptor, storage);
        const auto group_entry = selector_index->groups_by_ordinal.find(
            group_ordinal);
        if (group_entry == selector_index->groups_by_ordinal.end()) {
            throw LedgerError(
                p + ".owner points outside descriptor group/tile range");
        }
        const PipelineGroup& group =
            descriptor.groups[group_entry->second];
        if (tile_ordinal < 1 || tile_ordinal > group.tile_axis.tile_count) {
            throw LedgerError(
                p + ".owner points outside descriptor group/tile range");
        }
        const std::uint64_t global_tile_ordinal =
            group.tile_axis.global_tile_first + tile_ordinal - 1;
        if (slot_index != (global_tile_ordinal - 1) % 2) {
            throw LedgerError(
                p + ".slot_index disagrees with two-slot tile parity");
        }
        const std::optional<std::uint64_t> matching_run_index =
            descriptor_run_at(*selector_index, group_ordinal, rank,
                              tile_ordinal);
        if (role_source) {
            if (!matching_run_index.has_value()
                    || rank != descriptor.endpoint_runs[*matching_run_index]
                        .rank
                    || std::find(descriptor.endpoint_runs[*matching_run_index]
                                     .shard_ordinals.begin(),
                                 descriptor.endpoint_runs[*matching_run_index]
                                     .shard_ordinals.end(),
                                 *shard)
                        == descriptor.endpoint_runs[*matching_run_index]
                            .shard_ordinals.end()) {
                throw LedgerError(
                    p + " source shard is absent from the owner's endpoint run");
            }
        } else {
            if (rank != ident.rank) {
                throw LedgerError(
                    p + " target slot rank disagrees with descriptor target "
                       "rank");
            }
            for (const std::uint64_t target_shard_ordinal : targets) {
                if (target_shard_ordinal >= group.shards.size()) {
                    throw LedgerError(
                        p + " target consumption shards disagree with "
                           "descriptor tile");
                }
                const std::pair<std::uint64_t, PipelineShardTile> selected =
                    descriptor.selected_shard_tile(group_ordinal,
                                                   tile_ordinal - 1,
                                                   target_shard_ordinal);
                if (!selected.second.emit_endpoint
                        || selected.second.target_rank != ident.rank) {
                    throw LedgerError(
                        p + " target consumption shards disagree with "
                           "descriptor tile");
                }
            }
        }
    }
    if (consumed != !consumed_targets.empty()) {
        throw LedgerError(p + ".consumed must match consumed target shards");
    }
    if (!occupied && (consumed || !targets.empty() || !consumed_targets.empty())) {
        throw LedgerError(p + " empty slot cannot retain consume state");
    }
    if (role_source && (!targets.empty() || !consumed_targets.empty()
                            || consumed)) {
        throw LedgerError(p + " source slot cannot carry target consume state");
    }
    if (occupied && role_target && targets.empty()) {
        throw LedgerError(p + " occupied target slot needs consumption authority");
    }
    Json parsed = row;  // dict(row)
    Json targets_json = Json::array();
    for (const std::uint64_t value : targets) targets_json.push_back(value);
    Json consumed_json = Json::array();
    for (const std::uint64_t value : consumed_targets) {
        consumed_json.push_back(value);
    }
    parsed["target_shards_to_consume"] = std::move(targets_json);
    parsed["consumed_target_shards"] = std::move(consumed_json);
    return parsed;
}

Json slot_key_of(const Json& row) {
    return Json::array({row["buffer_domain_id"], row["role"], row["rank"],
                        row["shard_ordinal"], row["slot_index"]});
}

Json slot_sort_key_of(const Json& key) {
    Json shard = key[3].is_null() ? Json(std::int64_t(-1)) : key[3];
    return Json::array({key[0], key[1], key[2], std::move(shard), key[4]});
}

// pipeline_ledger.py _parse_row_patches, specialized per row kind.  The
// returned JSON is the downstream representation [[key, before, after], ...].
Json parse_row_patches(
    const Json& value, const std::string& path,
    const std::function<Json(const Json&, const std::string&)>& parse_row,
    const std::function<Json(const Json&)>& key_of,
    const std::function<Json(const Json&)>& sort_key_of) {
    if (!value.is_array()) throw LedgerError(path + " must be an array");
    Json parsed = Json::array();
    bool has_previous = false;
    Json previous_sort;
    for (std::size_t index = 0; index < value.size(); ++index) {
        const std::string p = path_index(path, index);
        const Json& raw = value[index];
        const Json& patch = read_mapping(raw, p);
        require_exact_keys(patch, p, patch_keys());
        const Json& before_raw = patch["before"];
        const Json& after_raw = patch["after"];
        if (before_raw.is_null() && after_raw.is_null()) {
            throw LedgerError(p + " cannot have both before and after null");
        }
        Json before;
        Json after;
        if (!before_raw.is_null()) {
            before = parse_row(before_raw, p + ".before");
        }
        if (!after_raw.is_null()) {
            after = parse_row(after_raw, p + ".after");
        }
        Json key;
        if (!before.is_null() && !after.is_null()) {
            const Json before_key = key_of(before);
            const Json after_key = key_of(after);
            if (!python_equal(before_key, after_key)) {
                throw LedgerError(p + " changes a row's stable key");
            }
            if (typed_equal(before, after)) {
                throw LedgerError(p + " is a no-op patch");
            }
            key = before_key;
        } else {
            key = key_of(before.is_null() ? after : before);
        }
        const Json sort_key = sort_key_of(key);
        if (has_previous && json_tuple_cmp(sort_key, previous_sort) <= 0) {
            throw LedgerError(path + " must be sorted and duplicate-free");
        }
        has_previous = true;
        previous_sort = sort_key;
        parsed.push_back(Json::array({key, before, after}));
    }
    return parsed;
}

}  // namespace

// --------------------------------------------------- slot patch union -----

Json parse_v4_slot_patches(Json value, const std::string& path,
                           const DescriptorView& descriptors,
                           const SelectorView& selector_indexes) {
    if (!value.is_array()) throw LedgerError(path + " must be an array");
    Json out = Json::array();
    bool has_previous = false;
    Json previous_sort;
    for (std::size_t index = 0; index < value.size(); ++index) {
        const std::string p = path_index(path, index);
        const Json& raw = value[index];
        bool is_target_delta = false;
        if (raw.is_object()) {
            const auto form = raw.find("form");
            is_target_delta = form != raw.end() && form->is_string()
                && form->get_ref<const std::string&>() == "target_delta";
        }
        Json item;
        Json sort_key;
        if (is_target_delta) {
            const Json& delta = read_mapping(raw, p);
            require_exact_keys(delta, p, target_delta_keys());
            const Json& key_row = read_mapping(delta["key"], p + ".key");
            require_exact_keys(key_row, p + ".key", target_key_keys());
            const std::string domain = read_string(
                key_row["buffer_domain_id"], p + ".key.buffer_domain_id");
            const Json& role = key_row["role"];
            const std::uint64_t rank = read_rank(key_row["rank"],
                                                 p + ".key.rank");
            const std::optional<std::uint64_t> shard = read_nullable_uint(
                key_row["shard_ordinal"], p + ".key.shard_ordinal");
            const std::uint64_t slot_index = read_uint(
                key_row["slot_index"], p + ".key.slot_index", 1);
            const bool role_is_target = role.is_string()
                && role.get_ref<const std::string&>() == "target";
            if (!role_is_target || shard.has_value()) {
                throw LedgerError(
                    p + ".key must identify a target slot with null shard");
            }
            const std::uint64_t before_generation = read_uint(
                delta["before_slot_generation"], p + ".before_slot_generation");
            const std::uint64_t predecessor_seq = read_uint(
                delta["predecessor_logical_seq"],
                p + ".predecessor_logical_seq");
            const std::uint64_t predecessor_count = read_uint(
                delta["predecessor_consumed_count"],
                p + ".predecessor_consumed_count");
            const Json& operations_wire = delta["operations"];
            if (!operations_wire.is_array()) {
                throw LedgerError(p + ".operations must be an array");
            }
            Json operations = Json::array();
            for (std::size_t op_index = 0; op_index < operations_wire.size();
                 ++op_index) {
                const std::string q = p + ".operations["
                    + std::to_string(op_index) + "]";
                const Json& raw_op = operations_wire[op_index];
                if (!raw_op.is_object()) {
                    throw LedgerError(q + " must be an object");
                }
                const auto op_field = raw_op.find("op");
                const std::string op_kind = op_field == raw_op.end()
                    ? read_string(Json(), q + ".op")
                    : read_string(*op_field, q + ".op");
                const auto generation_field = raw_op.find("slot_generation");
                const std::uint64_t generation = generation_field == raw_op.end()
                    ? read_uint(Json(), q + ".slot_generation")
                    : read_uint(*generation_field, q + ".slot_generation");
                Json operation = Json::object();
                if (op_kind == "initialize") {
                    static const std::vector<const char*> allowed = {
                        "op", "slot_generation"};
                    require_exact_keys(raw_op, q, allowed);
                    operation["op"] = op_kind;
                    operation["slot_generation"] = generation;
                } else if (op_kind == "claim") {
                    static const std::vector<const char*> allowed = {
                        "op", "slot_generation", "owner", "target_shards"};
                    require_exact_keys(raw_op, q, allowed);
                    const Json& owner_raw = read_mapping(raw_op["owner"],
                                                         q + ".owner");
                    require_exact_keys(owner_raw, q + ".owner", owner_keys());
                    Json owner = owner_raw;
                    owner["segment_id"] = read_string(owner_raw["segment_id"],
                                                      q + ".owner.segment_id");
                    static const char* owner_uints[] = {
                        "pipeline_generation", "stage_epoch",
                        "staging_allocation_id", "group_ordinal",
                        "tile_ordinal"};
                    for (const char* field : owner_uints) {
                        owner[field] = read_uint(owner_raw[field],
                                                 q + ".owner." + field);
                    }
                    const Json& shards_wire = raw_op["target_shards"];
                    if (!shards_wire.is_array()) {
                        throw LedgerError(q + ".target_shards must be an array");
                    }
                    Json target_shards = Json::array();
                    for (std::size_t i = 0; i < shards_wire.size(); ++i) {
                        target_shards.push_back(read_uint(
                            shards_wire[i], q + ".target_shards[]"));
                    }
                    for (std::size_t i = 1; i < target_shards.size(); ++i) {
                        if (json_int_value(target_shards[i - 1])
                                >= json_int_value(target_shards[i])) {
                            throw LedgerError(
                                q + ".target_shards must be sorted unique");
                        }
                    }
                    operation["op"] = op_kind;
                    operation["slot_generation"] = generation;
                    operation["owner"] = std::move(owner);
                    operation["target_shards"] = std::move(target_shards);
                } else if (op_kind == "consume") {
                    static const std::vector<const char*> allowed = {
                        "op", "slot_generation", "cause", "add_shards",
                        "cleared"};
                    require_exact_keys(raw_op, q, allowed);
                    const Json& cause_row = read_mapping(raw_op["cause"],
                                                         q + ".cause");
                    static const std::vector<const char*> cause_keys = {
                        "terminal_rank", "terminal_store_id",
                        "consume_run_index"};
                    require_exact_keys(cause_row, q + ".cause", cause_keys);
                    Json cause = Json::object();
                    cause["terminal_rank"] = read_rank(
                        cause_row["terminal_rank"], q + ".cause.terminal_rank");
                    cause["terminal_store_id"] = read_uint(
                        cause_row["terminal_store_id"],
                        q + ".cause.terminal_store_id");
                    cause["consume_run_index"] = read_uint(
                        cause_row["consume_run_index"],
                        q + ".cause.consume_run_index");
                    const Json& additions_wire = raw_op["add_shards"];
                    if (!additions_wire.is_array()) {
                        throw LedgerError(q + ".add_shards must be an array");
                    }
                    Json additions = Json::array();
                    for (std::size_t i = 0; i < additions_wire.size(); ++i) {
                        additions.push_back(read_uint(additions_wire[i],
                                                      q + ".add_shards[]"));
                    }
                    for (std::size_t i = 1; i < additions.size(); ++i) {
                        if (json_int_value(additions[i - 1])
                                >= json_int_value(additions[i])) {
                            throw LedgerError(
                                q + ".add_shards must be sorted unique");
                        }
                    }
                    operation["op"] = op_kind;
                    operation["slot_generation"] = generation;
                    operation["cause"] = std::move(cause);
                    operation["add_shards"] = std::move(additions);
                    operation["cleared"] = read_bool(raw_op["cleared"],
                                                     q + ".cleared");
                } else if (op_kind == "request_close_clear"
                           || op_kind == "erase") {
                    static const std::vector<const char*> allowed = {
                        "op", "slot_generation"};
                    require_exact_keys(raw_op, q, allowed);
                    operation["op"] = op_kind;
                    operation["slot_generation"] = generation;
                } else {
                    throw LedgerError(q + ".op is unknown");
                }
                operations.push_back(std::move(operation));
            }
            if (operations.empty()) {
                throw LedgerError(p + ".operations must be nonempty");
            }
            item = Json::object();
            item["form"] = "target_delta";
            item["key"] = Json::array({domain, role, rank, nullptr, slot_index});
            item["before_slot_generation"] = before_generation;
            item["predecessor_logical_seq"] = predecessor_seq;
            item["predecessor_consumed_count"] = predecessor_count;
            item["operations"] = std::move(operations);
            sort_key = Json::array({domain, role, rank, std::int64_t(-1),
                                    slot_index});
        } else {
            const Json patch = parse_row_patches(
                Json::array({raw}), p,
                [&](const Json& row_value, const std::string& q) {
                    return parse_slot_row(row_value, q, descriptors,
                                          selector_indexes);
                },
                slot_key_of, slot_sort_key_of)[0];
            const Json& key = patch[0];
            const Json& before = patch[1];
            const Json& after = patch[2];
            const bool before_bad = !before.is_null()
                && before["role"].get_ref<const std::string&>() != "source";
            const bool after_bad = !after.is_null()
                && after["role"].get_ref<const std::string&>() != "source";
            if (before_bad || after_bad) {
                throw LedgerError(
                    p + " v4 target slots must use target_delta form");
            }
            item = Json::array({"source", key, before, after});
            sort_key = slot_sort_key_of(key);
        }
        if (has_previous && json_tuple_cmp(sort_key, previous_sort) <= 0) {
            throw LedgerError(path + " must be sorted and duplicate-free");
        }
        has_previous = true;
        previous_sort = sort_key;
        out.push_back(std::move(item));
    }
    return out;
}

// -------------------------------------------------------- credit rows -----

namespace {

Json parse_credit_row(const Json& raw, const std::string& p) {
    const Json& row = read_mapping(raw, p);
    require_exact_keys(row, p, credit_keys());
    const std::uint64_t capacity = read_uint(row["capacity"], p + ".capacity");
    const std::uint64_t available = read_uint(row["available"], p + ".available");
    if (capacity != 2 || available > capacity) {
        throw LedgerError(p + " requires capacity=2 and available in [0,2]");
    }
    Json parsed = row;
    parsed["buffer_domain_id"] = read_string(row["buffer_domain_id"],
                                             p + ".buffer_domain_id");
    parsed["source_rank"] = read_rank(row["source_rank"], p + ".source_rank");
    parsed["shard_ordinal"] = read_uint(row["shard_ordinal"],
                                        p + ".shard_ordinal");
    parsed["outstanding_ack_count"] = read_uint(
        row["outstanding_ack_count"], p + ".outstanding_ack_count");
    return parsed;
}

}  // namespace

Json parse_credit_patches(Json value, const std::string& path) {
    return parse_row_patches(
        value, path, parse_credit_row,
        [](const Json& row) {
            return Json::array({row["buffer_domain_id"], row["source_rank"],
                                row["shard_ordinal"]});
        },
        [](const Json& key) { return key; });
}

// ---------------------------------------------------- obligation rows -----

namespace {

Json parse_obligation_row(const Json& raw, const std::string& p,
                          const DescriptorView& descriptors) {
    const Json& row = read_mapping(raw, p);
    require_exact_keys(row, p, obligation_keys());
    const std::string segment = read_string(row["segment_id"],
                                            p + ".segment_id");
    if (descriptors.find(segment) == nullptr) {
        throw LedgerError(p + " references unknown segment");
    }
    const std::uint64_t count = read_uint(row["remaining_endpoint_count"],
                                          p + ".remaining_endpoint_count");
    const Json& state = row["state"];
    const bool state_ok = state.is_string()
        && (state.get_ref<const std::string&>() == "pending"
            || state.get_ref<const std::string&>() == "active");
    if (count == 0 || !state_ok) {
        throw LedgerError(
            p + " must be pending/active with positive remaining count");
    }
    Json parsed = row;
    parsed["obligation_id"] = read_string(row["obligation_id"],
                                          p + ".obligation_id");
    parsed["rank"] = read_rank(row["rank"], p + ".rank");
    parsed["first_store_id"] = read_uint(row["first_store_id"],
                                         p + ".first_store_id");
    return parsed;
}

}  // namespace

Json parse_obligation_patches(Json value, const std::string& path,
                              const DescriptorView& descriptors) {
    return parse_row_patches(
        value, path,
        [&](const Json& row_value, const std::string& q) {
            return parse_obligation_row(row_value, q, descriptors);
        },
        [](const Json& row) { return row["obligation_id"]; },
        [](const Json& key) { return Json::array({key}); });
}

// ------------------------------------------------------ cursor patches ----

namespace {

bool completed_range_contains(const std::vector<std::pair<
        std::uint64_t, std::uint64_t>>& ranges, std::uint64_t ordinal) {
    const auto position = std::upper_bound(
        ranges.begin(), ranges.end(),
        std::make_pair(ordinal, U64_MAX));
    if (position == ranges.begin()) return false;
    const auto& candidate = *(position - 1);
    return ordinal < candidate.second;
}

Json parse_cursor_patch_run(const Json& raw_run, const std::string& path,
                            const PipelineDescriptor& descriptor,
                            const std::string& segment,
                            std::uint64_t run_index) {
    const Json& run = read_mapping(raw_run, path);
    require_exact_keys(run, path, cursor_run_keys());
    const PipelineEndpointRun& expected =
        descriptor.endpoint_runs.at(run_index);
    const std::uint64_t rank = read_rank(run["rank"], path + ".rank");
    const std::uint64_t group = read_uint(run["group_ordinal"],
                                          path + ".group_ordinal");
    const std::uint64_t first_tile = read_uint(run["first_tile"],
                                               path + ".first_tile");
    const std::uint64_t tile_count = read_uint(run["tile_count"],
                                               path + ".tile_count");
    if (tile_count == 0 || rank != expected.rank
            || group != expected.group_ordinal
            || first_tile != expected.first_tile
            || tile_count != expected.tile_count) {
        throw LedgerError(
            path + " does not match descriptor endpoint-run order/extent");
    }
    const std::uint64_t completed = read_uint(run["completed_endpoint_count"],
                                              path + ".completed_endpoint_count");
    const std::uint64_t cancelled = read_uint(
        run["cancelled_endpoint_count"], path + ".cancelled_endpoint_count");
    const std::uint64_t remaining = read_uint(
        run["remaining_endpoint_count"], path + ".remaining_endpoint_count");
    const bool in_flight = read_bool(run["in_flight"], path + ".in_flight");
    const std::optional<std::uint64_t> next_store = read_nullable_uint(
        run["next_store_id"], path + ".next_store_id");
    const std::optional<std::uint64_t> initial_parent = read_nullable_uint(
        run["initial_parent_store_id"], path + ".initial_parent_store_id");
    const std::uint64_t tile_offset = read_uint(run["tile_offset"],
                                                path + ".tile_offset");
    const std::uint64_t shard_index = read_uint(run["shard_index"],
                                                path + ".shard_index");
    const std::uint64_t leg_index = read_uint(run["leg_index"],
                                              path + ".leg_index");
    const std::uint64_t shard_count = expected.shard_ordinals.size();
    const std::uint64_t leg_count = expected.legs.size();
    const std::uint64_t total_count = tile_count * shard_count * leg_count;
    if (total_count == 0 || tile_offset > tile_count) {
        throw LedgerError(path + " has an invalid endpoint cursor extent");
    }
    if (tile_offset == tile_count) {
        if (shard_index != 0 || leg_index != 0 || next_store.has_value()) {
            throw LedgerError(
                path + " exhausted cursor position must have null next store");
        }
    } else if (shard_index >= shard_count || leg_index >= leg_count) {
        throw LedgerError(path + " cursor position exceeds run axes");
    }
    const std::optional<std::uint64_t> expected_parent =
        expected.initial_parent_json_id.has_value()
        ? std::optional<std::uint64_t>(
            *expected.initial_parent_json_id + 1)
        : std::nullopt;
    if (initial_parent != expected_parent) {
        throw LedgerError(
            path + ".initial_parent_store_id disagrees with descriptor");
    }
    const Json& ooo_wire = run["completed_out_of_order"];
    if (!ooo_wire.is_array()) {
        throw LedgerError(path + ".completed_out_of_order must be an array");
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> completed_ranges;
    if (completed != 0) {
        completed_ranges.emplace_back(std::uint64_t{0}, completed);
    }
    std::uint64_t out_of_order_count = 0;
    std::optional<std::uint64_t> last_end;
    for (std::size_t index = 0; index < ooo_wire.size(); ++index) {
        const std::string p = path + ".completed_out_of_order["
            + std::to_string(index) + "]";
        const Json& item = read_mapping(ooo_wire[index], p);
        static const std::vector<const char*> ooo_keys = {
            "first_ordinal", "end_ordinal_exclusive"};
        require_exact_keys(item, p, ooo_keys);
        const std::uint64_t begin = read_uint(item["first_ordinal"],
                                              p + ".first_ordinal");
        const std::uint64_t end = read_uint(item["end_ordinal_exclusive"],
                                            p + ".end_ordinal_exclusive");
        if (begin >= end || begin < completed || end > total_count
                || (last_end.has_value() && begin < *last_end)) {
            throw LedgerError(
                p + " is empty, overlaps, or precedes completed prefix");
        }
        last_end = end;
        out_of_order_count += end - begin;
        completed_ranges.emplace_back(begin, end);
    }
    if (completed + out_of_order_count + cancelled + remaining
            != total_count) {
        throw LedgerError(
            path + " completed/out-of-order/remaining counts do not cover its "
                   "endpoint run");
    }
    const Json& endpoints_wire = run["active_endpoints"];
    if (!endpoints_wire.is_array()) {
        throw LedgerError(path + ".active_endpoints must be an array");
    }
    Json parsed_endpoints = Json::array();
    std::set<std::uint64_t> seen_store_ids;
    std::set<std::uint64_t> seen_transfer_ids;
    for (std::size_t index = 0; index < endpoints_wire.size(); ++index) {
        const std::string p = path + ".active_endpoints["
            + std::to_string(index) + "]";
        const Json& item = read_mapping(endpoints_wire[index], p);
        require_exact_keys(item, p, active_endpoint_keys());
        const std::uint64_t store_id = read_uint(item["store_id"],
                                                 p + ".store_id");
        if (!seen_store_ids.insert(store_id).second) {
            throw LedgerError(path + " duplicates active store_id");
        }
        const std::uint64_t active_tile = read_uint(item["tile_ordinal"],
                                                    p + ".tile_ordinal");
        const std::uint64_t active_shard = read_uint(item["shard_ordinal"],
                                                     p + ".shard_ordinal");
        const std::string endpoint = read_string(item["endpoint"],
                                                 p + ".endpoint");
        const std::uint64_t transfer_id = read_uint(item["transfer_id"],
                                                    p + ".transfer_id");
        if (!seen_transfer_ids.insert(transfer_id).second) {
            throw LedgerError(path + " duplicates active transfer_id");
        }
        read_nullable_uint(item["flow_id"], p + ".flow_id");
        read_bool(item["network_done"], p + ".network_done");
        read_bool(item["hbm_done"], p + ".hbm_done");
        const std::int64_t local_tile = static_cast<std::int64_t>(
            active_tile - expected.first_tile);
        const auto shard_position_it = std::find(
            expected.shard_ordinals.begin(), expected.shard_ordinals.end(),
            active_shard);
        if (shard_position_it == expected.shard_ordinals.end()) {
            throw LedgerError(
                p + ".shard_ordinal is absent from its endpoint run");
        }
        const std::uint64_t shard_position = static_cast<std::uint64_t>(
            shard_position_it - expected.shard_ordinals.begin());
        std::optional<std::size_t> leg_position;
        for (std::size_t i = 0; i < expected.legs.size(); ++i) {
            if (expected.legs[i].endpoint == endpoint) {
                leg_position = i;
                break;
            }
        }
        if (local_tile < 0
                || local_tile >= static_cast<std::int64_t>(expected.tile_count)
                || !leg_position.has_value()) {
            throw LedgerError(p + " endpoint position is outside its run");
        }
        const std::uint64_t ordinal =
            static_cast<std::uint64_t>(local_tile) * shard_count * leg_count
            + shard_position * leg_count
            + static_cast<std::uint64_t>(*leg_position);
        if (completed_range_contains(completed_ranges, ordinal)) {
            throw LedgerError(p + " is simultaneously complete and active");
        }
        const std::uint64_t expected_id = descriptor.endpoint_id(
            run_index, active_tile, active_shard, *leg_position);
        if (store_id != expected_id + 1) {
            throw LedgerError(
                p + ".store_id disagrees with endpoint ID + 1");
        }
        const Json& owner = read_mapping(item["owner"], p + ".owner");
        require_exact_keys(owner, p + ".owner", owner_keys());
        const PipelineIdentity& ident = descriptor.identity;
        // Python compares raw owner values with == (int/float coercive).
        if (!(python_equal(owner["segment_id"], Json(segment))
                && python_equal(owner["pipeline_generation"],
                                Json(ident.generation))
                && python_equal(owner["stage_epoch"], Json(ident.stage_epoch))
                && python_equal(owner["staging_allocation_id"],
                                Json(ident.staging_allocation_id))
                && python_equal(owner["group_ordinal"],
                                Json(expected.group_ordinal))
                && python_equal(owner["tile_ordinal"], Json(active_tile)))) {
            throw LedgerError(
                p + ".owner disagrees with its descriptor endpoint");
        }
        if (expected.legs[*leg_position].endpoint != endpoint) {
            throw LedgerError(p + ".endpoint disagrees with run leg");
        }
        if (endpoint != "body_marker") {
            const std::pair<std::uint64_t, PipelineShardTile> selected =
                descriptor.selected_shard_tile(expected.group_ordinal,
                                               active_tile - 1, active_shard);
            if (!selected.second.emit_endpoint) {
                throw LedgerError(
                    p + " maps to an inactive stream-bytes endpoint");
            }
        }
        parsed_endpoints.push_back(item);
    }
    if (in_flight != !endpoints_wire.empty()) {
        throw LedgerError(path + ".in_flight must match active_endpoints");
    }
    if (tile_offset < tile_count) {
        const std::uint64_t tile_ordinal = expected.first_tile + tile_offset;
        const std::uint64_t shard = expected.shard_ordinals[shard_index];
        const PipelineEndpointLegRun& leg = expected.legs[leg_index];
        if (leg.endpoint != "body_marker") {
            const std::pair<std::uint64_t, PipelineShardTile> selected =
                descriptor.selected_shard_tile(expected.group_ordinal,
                                               tile_ordinal - 1, shard);
            if (!selected.second.emit_endpoint) {
                throw LedgerError(
                    path + " cursor position points to an inactive endpoint");
            }
        }
        const std::uint64_t expected_next = descriptor.endpoint_id(
            run_index, tile_ordinal, shard, leg_index) + 1;
        if (next_store != expected_next) {
            throw LedgerError(
                path + ".next_store_id disagrees with cursor position");
        }
    }
    Json parsed_run = run;
    parsed_run["completed_out_of_order"] = ooo_wire;
    parsed_run["active_endpoints"] = std::move(parsed_endpoints);
    return parsed_run;
}

}  // namespace

Json parse_cursor_patch_one(Json raw, const std::string& p,
                            const DescriptorView& descriptors,
                            CursorSegmentCursor* cursor) {
    const Json& patch = read_mapping(raw, p);
    static const std::vector<const char*> patch_key_set = {
        "segment_id", "cursor_version", "retired", "runs"};
    require_exact_keys(patch, p, patch_key_set);
    const std::string segment = read_string(patch["segment_id"],
                                            p + ".segment_id");
    const DescriptorPtr* found = descriptors.find(segment);
    if (found == nullptr) {
        throw LedgerError(p + " references unknown descriptor segment");
    }
    const PipelineDescriptor& descriptor = **found;
    const std::uint64_t version = read_uint(patch["cursor_version"],
                                            p + ".cursor_version");
    const bool retired = read_bool(patch["retired"], p + ".retired");
    const Json& run_rows = patch["runs"];
    if (!run_rows.is_array()) {
        throw LedgerError(p + ".runs must be an array");
    }
    Json parsed_runs = Json::array();
    std::optional<std::uint64_t> previous_index;
    static const std::vector<const char*> run_item_keys = {
        "run_index", "value"};
    for (std::size_t index = 0; index < run_rows.size(); ++index) {
        const std::string q = p + ".runs[" + std::to_string(index) + "]";
        const Json& item = read_mapping(run_rows[index], q);
        require_exact_keys(item, q, run_item_keys);
        const std::uint64_t ordinal = read_uint(item["run_index"],
                                                q + ".run_index");
        if (ordinal >= descriptor.endpoint_runs.size()) {
            throw LedgerError(q + ".run_index exceeds descriptor runs");
        }
        const Json& raw_value = item["value"];
        Json parsed_value;
        if (raw_value.is_null()) {
            parsed_value = nullptr;
        } else {
            parsed_value = parse_cursor_patch_run(raw_value, q + ".value",
                                                  descriptor, segment,
                                                  ordinal);
        }
        if (parsed_value.is_null() && !retired) {
            throw LedgerError(
                q + ".value may be null only during descriptor retirement");
        }
        if (!parsed_value.is_null() && retired) {
            throw LedgerError(
                q + ".value must be null on descriptor retirement");
        }
        if (previous_index.has_value() && ordinal <= *previous_index) {
            throw LedgerError(p + ".runs must be sorted and duplicate-free");
        }
        previous_index = ordinal;
        parsed_runs.push_back(Json::array({ordinal, std::move(parsed_value)}));
    }
    if (retired) {
        const std::uint64_t total = descriptor.endpoint_runs.size();
        bool tombstoned = parsed_runs.size() == total;
        if (tombstoned && !parsed_runs.empty()) {
            const std::uint64_t first = static_cast<std::uint64_t>(
                json_int_value(parsed_runs[0][0]));
            const std::uint64_t last = static_cast<std::uint64_t>(
                json_int_value(parsed_runs[parsed_runs.size() - 1][0]));
            tombstoned = first == 0 && last == total - 1;
        }
        if (tombstoned) {
            for (const Json& entry : parsed_runs) {
                if (!entry[1].is_null()) tombstoned = false;
            }
        }
        if (!tombstoned) {
            throw LedgerError(
                p + " retirement must tombstone every descriptor cursor run");
        }
    }
    if (cursor->has_previous && segment <= cursor->previous_segment) {
        throw LedgerError("cursor_patches must be sorted and duplicate-free");
    }
    cursor->has_previous = true;
    cursor->previous_segment = segment;
    Json out = Json::object();
    out["segment_id"] = segment;
    out["cursor_version"] = version;
    out["retired"] = retired;
    out["runs"] = std::move(parsed_runs);
    return out;
}

// ------------------------------------------------------------- runs -------

Json parse_one_pipeline_run(Json raw, const std::string& p,
                            const std::string& section_path,
                            const DescriptorView& descriptors,
                            const SelectorView& selector_indexes,
                            RunSeqCursor* cursor) {
    const Json& row = read_mapping(raw, p);
    require_exact_keys(row, p, run_keys());
    const std::uint64_t seq = read_uint(row["first_logical_seq"],
                                        p + ".first_logical_seq");
    if (seq == 0) {
        throw LedgerError(p + ".first_logical_seq must start at 1");
    }
    read_string(row["pattern_id"], p + ".pattern_id");
    const std::uint64_t repeat = read_uint(row["repeat_count"],
                                           p + ".repeat_count");
    if (repeat != 1) {
        throw LedgerError(p + ".repeat_count must equal 1");
    }
    const Json& transition = row["transition_kind"];
    if (!in_strings(transition, kTransitions,
                    sizeof kTransitions / sizeof kTransitions[0])) {
        throw LedgerError(p + ".transition_kind is unknown");
    }
    const std::string segment = read_string(row["descriptor_segment_id"],
                                            p + ".descriptor_segment_id");
    const DescriptorPtr* descriptor_ptr = descriptors.find(segment);
    if (descriptor_ptr == nullptr) {
        throw LedgerError(p + " references undeclared descriptor '" + segment
                          + "'");
    }
    const PipelineDescriptor& descriptor = **descriptor_ptr;
    const Json& selector = row["endpoint_selector"];
    const Json& store = row["store_id_first"];
    const Json& store_stride = row["store_id_stride"];
    const Json& legacy = row["legacy_context"];
    const Json* endpoint_fields[4] = {&selector, &store, &store_stride,
                                      &legacy};
    bool all_null = true;
    bool any_null = false;
    for (const Json* field : endpoint_fields) {
        if (field->is_null()) {
            any_null = true;
        } else {
            all_null = false;
        }
    }
    bool have_selector = false;
    SelectorResolution resolution;
    std::uint64_t tile_first = 0;
    std::uint64_t shard_first = 0;
    std::uint64_t rank_first = 0;
    std::optional<std::uint64_t> endpoint_ordinal;
    std::optional<std::uint64_t> selected_global_ordinal;
    std::optional<PipelineShardTile> selected_shard_facts;
    std::optional<std::uint64_t> endpoint_store_id;
    if (all_null) {
        static const char* affine_fields[] = {
            "tile_first", "tile_stride", "rank_first", "rank_stride",
            "shard_first", "shard_stride"};
        for (const char* field : affine_fields) {
            if (!row[field].is_null()) {
                throw LedgerError(p + "." + field
                                  + " must be null on descriptor transition");
            }
        }
        if (!row["resource_leg"].is_null()) {
            throw LedgerError(
                p + ".resource_leg must be null on descriptor transition");
        }
    } else if (any_null) {
        throw LedgerError(
            p + " endpoint selector/store/context fields must be all null or "
               "all present");
    } else {
        have_selector = true;
        const Json& selector_mapping = read_mapping(
            selector, p + ".endpoint_selector");
        require_exact_keys(selector_mapping, p + ".endpoint_selector",
                           selector_keys());
        SelectorIndex storage;
        const SelectorIndex* selector_index = resolve_selector_index(
            selector_indexes, segment, descriptor, storage);
        rank_first = read_rank(row["rank_first"], p + ".rank_first");
        resolution = resolve_selector(descriptor, selector_mapping,
                                      *selector_index, rank_first);
        tile_first = read_uint(row["tile_first"], p + ".tile_first");
        shard_first = read_uint(row["shard_first"], p + ".shard_first");
        if (tile_first != resolution.tile || shard_first != resolution.shard
                || rank_first
                    != descriptor.endpoint_runs[resolution.run_index].rank) {
            throw LedgerError(p + " affine selector axes disagree");
        }
        static const char* stride_fields[] = {
            "tile_stride", "rank_stride", "shard_stride", "store_id_stride"};
        for (const char* field : stride_fields) {
            if (read_uint(row[field], p + "." + field) != 0) {
                throw LedgerError(p + "." + field
                                  + " must be 0 while repeat_count=1");
            }
        }
        const std::uint64_t store_id = read_uint(store, p + ".store_id_first");
        if (store_id != resolution.json_id + 1) {
            throw LedgerError(
                p + ".store_id_first must equal endpoint JSON ID + 1");
        }
        const Json& legacy_mapping = read_mapping(legacy,
                                                  p + ".legacy_context");
        static const std::vector<const char*> legacy_keys = {
            "request_id", "stage", "generation"};
        require_exact_keys(legacy_mapping, p + ".legacy_context",
                           legacy_keys);
        const PipelineLegacyContext& endpoint_context =
            descriptor.endpoint_runs[resolution.run_index].legacy_context;
        if (!(python_equal(legacy_mapping["request_id"],
                           Json(endpoint_context.request_id))
                && python_equal(legacy_mapping["stage"],
                                Json(endpoint_context.stage))
                && python_equal(legacy_mapping["generation"],
                                Json(endpoint_context.generation)))) {
            throw LedgerError(
                p + ".legacy_context disagrees with endpoint run");
        }
        const PipelineEndpointRun& endpoint_run =
            descriptor.endpoint_runs[resolution.run_index];
        const auto shard_position = std::find(
            endpoint_run.shard_ordinals.begin(),
            endpoint_run.shard_ordinals.end(), resolution.shard);
        if (shard_position == endpoint_run.shard_ordinals.end()) {
            throw LedgerError(p + " affine selector axes disagree");
        }
        endpoint_ordinal = (resolution.tile - endpoint_run.first_tile)
            * endpoint_run.shard_ordinals.size() * endpoint_run.legs.size()
            + static_cast<std::uint64_t>(shard_position
                - endpoint_run.shard_ordinals.begin())
                * endpoint_run.legs.size()
            + resolution.leg_index;
        endpoint_store_id = store_id;
    }
    const Json& resource_leg = row["resource_leg"];
    if (!resource_leg.is_null()
            && !in_strings(resource_leg, kResourceLegs,
                           sizeof kResourceLegs / sizeof kResourceLegs[0])) {
        throw LedgerError(p + ".resource_leg is unknown");
    }
    const std::uint64_t tick = read_uint(row["tick_first"], p + ".tick_first");
    if (read_uint(row["tick_stride"], p + ".tick_stride") != 0) {
        throw LedgerError(p + ".tick_stride must be 0");
    }
    const std::uint64_t cv0 = read_uint(row["cursor_version_start"],
                                        p + ".cursor_version_start");
    const std::uint64_t cv1 = read_uint(row["cursor_version_end"],
                                        p + ".cursor_version_end");
    if (cv1 < cv0 || cv1 - cv0 > 1) {
        throw LedgerError(
            p + " cursor version transition must be stable or increment by "
               "one");
    }
    const Json& prefix = row["resource_quantity_prefix"];
    if (!prefix.is_array()) {
        throw LedgerError(p + ".resource_quantity_prefix must be an array");
    }
    if (prefix.size() != 2) {
        throw LedgerError(
            p + ".resource_quantity_prefix length must equal repeat_count+1");
    }
    const Json& baseline = amount_mapping(
        prefix[0], p + ".resource_quantity_prefix[0]");
    const Json& quantity = amount_mapping(
        prefix[1], p + ".resource_quantity_prefix[1]");
    static const char* dims[] = {"ops", "bytes", "latency_ns"};
    for (const char* dim : dims) {
        const bool baseline_null = baseline[dim].is_null();
        const bool quantity_null = quantity[dim].is_null();
        if (baseline_null != quantity_null) {
            throw LedgerError(p + " resource prefix changes " + dim
                              + " nullability");
        }
        if (quantity_null) {
            if (!baseline_null) {
                throw LedgerError(p + " undefined " + dim
                                  + " baseline must be null");
            }
        } else {
            const Json& baseline_value = baseline[dim];
            const bool int_zero = json_is_int(baseline_value)
                && (baseline_value.is_number_unsigned()
                        ? baseline_value.get<std::uint64_t>() == 0
                        : baseline_value.get<std::int64_t>() == 0);
            if (!int_zero) {
                throw LedgerError(p + " defined " + dim
                                  + " baseline must be integer zero");
            }
        }
    }
    if (resource_leg.is_null()) {
        for (const char* dim : dims) {
            if (!quantity[dim].is_null()) {
                throw LedgerError(
                    p + " resource quantities require a resource_leg");
            }
        }
    }
    if (!resource_leg.is_null() && endpoint_ordinal.has_value()) {
        const std::pair<std::uint64_t, PipelineShardTile> selected =
            descriptor.selected_shard_tile(resolution.group,
                                           resolution.tile - 1,
                                           resolution.shard);
        selected_global_ordinal = selected.first;
        selected_shard_facts = selected.second;
        const Json expected = expected_endpoint_amount(
            descriptor, static_cast<std::size_t>(resolution.run_index),
            *endpoint_ordinal, selected.second);
        if (!typed_equal(quantity, expected)) {
            throw LedgerError(
                p + " resource quantity does not match descriptor endpoint");
        }
        const std::string& endpoint_name =
            descriptor.endpoint_runs[resolution.run_index]
                .legs[resolution.leg_index].endpoint;
        const bool leg_is_marker = resource_leg == "body_marker";
        if (leg_is_marker != (endpoint_name == "body_marker")) {
            throw LedgerError(
                p + ".resource_leg disagrees with descriptor endpoint kind");
        }
    }
    // The wire row is a discarded temporary (EnvelopeReader dispatch passes
    // it by value): move the three patch arrays into their parsers instead
    // of re-copying each subtree. All scalar reads from `row` are complete
    // by this point.
    Json slot_patches = parse_v4_slot_patches(
        std::move(raw["slot_patches"]), p + ".slot_patches", descriptors,
        selector_indexes);
    Json credit_patches = parse_credit_patches(
        std::move(raw["credit_patches"]), p + ".credit_patches");
    Json obligation_patches = parse_obligation_patches(
        std::move(raw["obligation_patches"]), p + ".obligation_patches",
        descriptors);
    const std::string transition_kind =
        transition.get_ref<const std::string&>();
    const bool descriptor_level =
        transition_kind == "descriptor_commit"
        || transition_kind == "service_done"
        || transition_kind == "request_close"
        || transition_kind == "settle_done"
        || transition_kind == "descriptor_retire"
        || (transition_kind == "consume_done" && !have_selector);
    if (descriptor_level && (have_selector || !resource_leg.is_null())) {
        throw LedgerError(
            p + " descriptor-level transition cannot bind an endpoint or "
               "resource");
    }
    if (transition_kind == "endpoint_done" && !have_selector) {
        throw LedgerError(
            p + " endpoint_done transition requires a concrete selector");
    }
    if (transition_kind != "descriptor_commit"
            && transition_kind != "service_done"
            && transition_kind != "request_close"
            && transition_kind != "settle_done"
            && transition_kind != "descriptor_retire"
            && !have_selector && transition_kind != "consume_done") {
        throw LedgerError(
            p + " endpoint transition requires a concrete selector");
    }
    if (have_selector && !selected_shard_facts.has_value()) {
        const std::pair<std::uint64_t, PipelineShardTile> selected =
            descriptor.selected_shard_tile(resolution.group,
                                           resolution.tile - 1,
                                           resolution.shard);
        selected_global_ordinal = selected.first;
        selected_shard_facts = selected.second;
    }
    if (cursor->has_previous && seq <= cursor->previous_seq) {
        throw LedgerError(section_path + " must preserve logical sequence "
                           "order");
    }
    cursor->has_previous = true;
    cursor->previous_seq = seq;
    Json parsed = Json::object();
    parsed["first_logical_seq"] = seq;
    parsed["descriptor_segment_id"] = segment;
    parsed["transition_kind"] = transition_kind;
    if (have_selector) {
        parsed["endpoint_selector"] = selector;
    } else {
        parsed["endpoint_selector"] = nullptr;
    }
    parsed["endpoint_run_index"] = have_selector
        ? Json(resolution.run_index) : Json();
    parsed["endpoint_leg_index"] = have_selector
        ? Json(resolution.leg_index) : Json();
    parsed["endpoint_ordinal"] = endpoint_ordinal.has_value()
        ? Json(*endpoint_ordinal) : Json();
    parsed["_selected_global_tile_ordinal"] = selected_global_ordinal
        .has_value() ? Json(*selected_global_ordinal) : Json();
    if (selected_shard_facts.has_value()) {
        Json tile = Json::object();
        tile["shard_ordinal"] = selected_shard_facts->shard_ordinal;
        tile["source_rank"] = selected_shard_facts->source_rank;
        tile["target_rank"] = selected_shard_facts->target_rank;
        tile["data_bytes"] = selected_shard_facts->data_bytes;
        tile["emit_endpoint"] = selected_shard_facts->emit_endpoint;
        parsed["_selected_shard_tile"] = std::move(tile);
    } else {
        parsed["_selected_shard_tile"] = nullptr;
    }
    parsed["endpoint_store_id"] = endpoint_store_id.has_value()
        ? Json(*endpoint_store_id) : Json();
    parsed["slot_patches"] = std::move(slot_patches);
    parsed["credit_patches"] = std::move(credit_patches);
    parsed["obligation_patches"] = std::move(obligation_patches);
    parsed["tick_first"] = tick;
    parsed["cursor_version_start"] = cv0;
    parsed["cursor_version_end"] = cv1;
    return parsed;
}

// --------------------------------------------------------- resources ------

Json parse_resource_row(Json raw, const std::string& p,
                        const DescriptorView& descriptors,
                        const SelectorView& selector_indexes,
                        std::optional<ResourceSortKey>* previous) {
    static const std::map<std::string, std::vector<const char*>> extras = {
        {"noc_flow", {"flow_id", "current_rate_Bpns",
                      "last_rate_update_time"}},
        {"hbm_job", {"kind", "start_tick"}},
        {"pool_job", {"port_index", "issue_seq", "latency_ready_ns"}}};
    static const std::map<std::string, std::vector<std::string>>
        allowed_states = {
            {"noc_flow", {"Active", "PropagatingTail", "Completed"}},
            {"hbm_job", {"issued", "in_service"}},
            {"pool_job", {"LatencyWaiting", "Streaming",
                          "FluidCompleteAwaitingCallback"}}};
    const Json& raw_mapping = read_mapping(raw, p);
    const auto kind_field = raw_mapping.find("resource_type");
    if (kind_field == raw_mapping.end() || !kind_field->is_string()
            || extras.find(kind_field->get_ref<const std::string&>())
                == extras.end()) {
        throw LedgerError(p + ".resource_type is unknown");
    }
    const std::string kind = kind_field->get_ref<const std::string&>();
    std::vector<const char*> allowed = resource_common_keys();
    for (const char* key : extras.at(kind)) allowed.push_back(key);
    const Json& row = raw_mapping;
    require_exact_keys(row, p, allowed);
    const Json& state = row["state"];
    const auto states = allowed_states.find(kind);
    bool state_ok = false;
    if (states != allowed_states.end() && state.is_string()) {
        for (const std::string& candidate : states->second) {
            if (state.get_ref<const std::string&>() == candidate) {
                state_ok = true;
                break;
            }
        }
    }
    if (!state_ok) {
        throw LedgerError(p + ".state is invalid for " + kind);
    }
    static const char* string_fields[] = {
        "work_id", "request_id", "operation", "leg", "progress_source",
        "service_unit"};
    for (const char* field : string_fields) {
        read_string(row[field], p + "." + field);
    }
    if (!row["stage"].is_string()) {
        throw LedgerError(p + ".stage must be a string");
    }
    if (row["service_unit"] != Json("bytes")) {
        throw LedgerError(p + ".service_unit must be bytes");
    }
    read_uint(row["generation"], p + ".generation");
    const std::uint64_t rank = read_rank(row["rank"], p + ".rank");
    const std::uint64_t node_id = read_uint(row["node_id"], p + ".node_id");
    read_uint(row["as_of_tick"], p + ".as_of_tick");
    read_nullable_uint(row["issued_tick"], p + ".issued_tick");
    if (row["progress_source"] != Json("authoritative")) {
        throw LedgerError(p + ".progress_source must be authoritative");
    }
    const Json& total = amount_mapping(row["total"], p + ".total");
    const Json& remaining = amount_mapping(row["remaining"], p + ".remaining");
    const Json& route_wire = row["route"];
    if (!route_wire.is_array()) {
        throw LedgerError(p + ".route must be an array");
    }
    for (std::size_t j = 0; j < route_wire.size(); ++j) {
        read_uint(route_wire[j], p + ".route[" + std::to_string(j) + "]");
    }
    const Json& segment = row["descriptor_segment_id"];
    const Json& selector = row["endpoint_selector"];
    bool bound = false;
    if (segment.is_null() != selector.is_null()) {
        throw LedgerError(
            p + " descriptor segment and endpoint selector must be both null "
               "or present");
    }
    if (!segment.is_null()) {
        bound = true;
        const std::string segment_id = read_string(
            segment, p + ".descriptor_segment_id");
        const DescriptorPtr* found = descriptors.find(segment_id);
        if (found == nullptr) {
            throw LedgerError(p + " references unknown descriptor");
        }
        const PipelineDescriptor& descriptor = **found;
        const Json& selector_mapping = read_mapping(
            selector, p + ".endpoint_selector");
        require_exact_keys(selector_mapping, p + ".endpoint_selector",
                           selector_keys());
        SelectorIndex storage;
        const SelectorIndex* selector_index = resolve_selector_index(
            selector_indexes, segment_id, descriptor, storage);
        const SelectorResolution resolution = resolve_selector(
            descriptor, selector_mapping, *selector_index, rank);
        const PipelineEndpointRun& run =
            descriptor.endpoint_runs[resolution.run_index];
        const PipelineLegacyContext& context = run.legacy_context;
        if (node_id != resolution.json_id + 1 || rank != run.rank) {
            throw LedgerError(
                p + " pipeline NodeStore mapping is inconsistent");
        }
        if (!(python_equal(row["request_id"], Json(context.request_id))
                && python_equal(row["stage"], Json(context.stage))
                && python_equal(row["generation"], Json(context.generation)))) {
            throw LedgerError(
                p + " pipeline context disagrees with endpoint legacy_context");
        }
        const std::string& leg = row["leg"].get_ref<const std::string&>();
        if (kind == "noc_flow" && leg.rfind("noc:", 0) != 0) {
            throw LedgerError(p + " NoC resource leg name is invalid");
        }
        if (kind == "hbm_job" && leg.rfind("hbm:", 0) != 0) {
            throw LedgerError(p + " HBM resource leg name is invalid");
        }
        if (kind == "pool_job" && leg.rfind("pool:", 0) != 0) {
            throw LedgerError(p + " pool resource leg name is invalid");
        }
    } else {
        const std::string& stage = row["stage"].get_ref<const std::string&>();
        const bool unattributed_noc = kind == "noc_flow"
            && python_equal(row["request_id"], Json("noc"))
            && stage.empty() && python_equal(row["generation"], Json(0));
        const bool external_background_hbm = kind == "hbm_job"
            && python_equal(row["request_id"], Json("external_background"))
            && stage.empty() && python_equal(row["generation"], Json(0));
        const std::string& stage_value =
            row["stage"].get_ref<const std::string&>();
        if (!(unattributed_noc || external_background_hbm)
                && stage_value != "prefill" && stage_value != "decode"
                && stage_value != "completion") {
            throw LedgerError(p + ".stage is invalid");
        }
    }
    if (bound) {
        const std::string& stage_value =
            row["stage"].get_ref<const std::string&>();
        if (stage_value != "prefill" && stage_value != "decode"
                && stage_value != "completion") {
            throw LedgerError(p + ".stage is invalid");
        }
    }
    if (kind != "noc_flow" && !route_wire.empty()) {
        throw LedgerError(p + " non-NoC route must be empty");
    }
    const std::string expected_work_id = leg_work_id(
        canonical_work_id(row["request_id"].get_ref<const std::string&>(),
                          row["stage"].get_ref<const std::string&>(),
                          read_uint(row["generation"], p + ".generation"),
                          rank, node_id),
        row["leg"].get_ref<const std::string&>());
    if (row["work_id"].get_ref<const std::string&>() != expected_work_id) {
        throw LedgerError(
            p + ".work_id is not canonical for its typed identity/leg");
    }
    if (kind != "hbm_job") {
        static const char* amount_names[] = {"total", "remaining"};
        for (const char* amount_name : amount_names) {
            const Json& amount = amount_name[0] == 't' ? total : remaining;
            if (!amount["ops"].is_null() || !amount["latency_ns"].is_null()) {
                throw LedgerError(std::string(p + ".") + amount_name + " for "
                                  + kind + " may define bytes only");
            }
        }
    }
    if (kind == "noc_flow") {
        const std::uint64_t flow_id = read_uint(row["flow_id"], p + ".flow_id");
        if (row["leg"] != Json("noc:" + std::to_string(flow_id))) {
            throw LedgerError(
                p + ".leg must identify its original flow_id");
        }
        if (!bound && !(python_equal(row["request_id"], Json("noc"))
                && row["stage"].get_ref<const std::string&>().empty()
                && python_equal(row["generation"], Json(0))
                && python_equal(row["node_id"], Json(flow_id)))) {
            throw LedgerError(
                p + " unattributed NoC identity must preserve noc/empty-stage/"
                   "flow-id form");
        }
        read_finite_nonnegative(row["current_rate_Bpns"],
                                p + ".current_rate_Bpns");
        read_finite_nonnegative(row["last_rate_update_time"],
                                p + ".last_rate_update_time");
        if (row["state"] == Json("Completed")
                && !python_equal(remaining["bytes"], Json(std::uint64_t{0}))) {
            throw LedgerError(
                p + " completed NoC tail must have zero remaining bytes");
        }
    } else if (kind == "hbm_job") {
        read_string(row["kind"], p + ".kind");
        const std::string& leg = row["leg"].get_ref<const std::string&>();
        if (python_equal(row["request_id"], Json("external_background"))
                && row["stage"].get_ref<const std::string&>().empty()
                && python_equal(row["generation"], Json(0))) {
            const std::string job_kind =
                row["kind"].get_ref<const std::string&>();
            if (job_kind != "COMM_READ" && job_kind != "COMM_WRITE") {
                throw LedgerError(
                    p + " external background HBM job kind must be a "
                       "transfer-sink COMM_READ/COMM_WRITE");
            }
            const std::size_t hash = leg.find('#');
            const std::string leg_rank = leg.substr(0, hash);
            const std::string leg_seq = hash == std::string::npos
                ? std::string() : leg.substr(hash + 1);
            const std::size_t colon = leg_rank.find(':');
            const std::string head = leg_rank.substr(0, colon);
            const std::string tail = colon == std::string::npos
                ? std::string() : leg_rank.substr(colon + 1);
            bool seq_ok = false;
            if (ascii_digits(leg_seq)) {
                std::uint64_t seq_value = 0;
                if (parse_decimal_u64(leg_seq, seq_value)) {
                    seq_ok = seq_value >= 1;
                } else {
                    seq_ok = true;  // beyond uint64: Python int(...) >= 1
                }
            }
            std::uint64_t rank_value = 0;
            const bool rank_ok = parse_decimal_u64(tail, rank_value)
                && python_equal(Json(rank_value), row["rank"]);
            const bool leg_ok = leg.rfind("hbm_external:", 0) == 0
                && seq_ok && head == "hbm_external" && ascii_digits(tail)
                && rank_ok;
            if (!leg_ok) {
                throw LedgerError(
                    p + ".leg must be the literal external background form "
                       "hbm_external:{rank}#{issue_seq} with the row's own "
                       "rank and a positive backend issue_seq");
            }
        } else if (row["leg"]
                != Json("hbm:" + row["kind"].get_ref<const std::string&>())) {
            throw LedgerError(
                p + ".leg must identify its HBM job kind");
        }
        read_uint(row["start_tick"], p + ".start_tick");
    } else {
        const std::uint64_t port = read_uint(row["port_index"],
                                             p + ".port_index");
        const std::uint64_t issue_seq = read_uint(row["issue_seq"],
                                                  p + ".issue_seq");
        if (row["leg"] != Json("pool:" + std::to_string(port) + ":"
                               + std::to_string(issue_seq))) {
            throw LedgerError(
                p + ".leg must identify its original pool job");
        }
        read_finite_nonnegative(row["latency_ready_ns"],
                                p + ".latency_ready_ns");
    }
    static const char* dims[] = {"ops", "bytes", "latency_ns"};
    for (const char* dim : dims) {
        const Json& total_value = total[dim];
        const Json& remaining_value = remaining[dim];
        if (total_value.is_null() != remaining_value.is_null()) {
            throw LedgerError(p + " changes " + dim + " applicability");
        }
        if (!total_value.is_null() && json_scalar_cmp(remaining_value,
                                                      total_value) > 0) {
            throw LedgerError(p + " remaining " + dim + " exceeds total");
        }
    }
    const ResourceSortKey sort_key = std::make_tuple(
        segment.is_null() ? std::string() : segment
            .get_ref<const std::string&>(),
        rank, node_id, row["leg"].get_ref<const std::string&>(), kind);
    if (previous->has_value() && !(sort_key > previous->value())) {
        throw LedgerError("active_resources must be sorted and duplicate-free");
    }
    *previous = sort_key;
    return row;
}

}  // namespace rowport

std::string canonical_envelope_bytes(const char* data, std::size_t length) {
    Json dom;
    try {
        dom = Json::parse(data, data + length);
    } catch (const Json::parse_error& error) {
        throw LedgerError(
            std::string("envelope cannot be parsed as canonical JSON at byte ")
            + std::to_string(error.byte) + ": " + error.what());
    }
    return rowport::canonical_json_dumps(dom);
}

}  // namespace jointledger
