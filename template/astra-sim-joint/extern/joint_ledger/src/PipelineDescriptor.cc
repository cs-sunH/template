// PipelineDescriptor.cc -- strict 1:1 port of joint/pipeline_descriptor.py
// (sh_test_mesh/workload/llama2_7b_inference/joint/pipeline_descriptor.py).
//
// Every validator, cross-check, checked-arithmetic operation and wire builder
// below mirrors the same-named Python function; arithmetic that Python runs in
// its unbounded integer domain uses unsigned/signed __int128 intermediates so
// exactly the same inputs are accepted or rejected.  All errors are
// jointledger::LedgerError carrying the offending path.
#include "joint_ledger/PipelineDescriptor.hh"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace jointledger {
namespace {

using u64 = std::uint64_t;
using i64 = std::int64_t;
using u128 = unsigned __int128;
using i128 = __int128;

constexpr u64 kU32Max = 0xFFFFFFFFULL;
constexpr i64 kI64Min = std::numeric_limits<i64>::min();
constexpr i64 kI64Max = std::numeric_limits<i64>::max();

// pipeline_descriptor.py module constants.
constexpr const char* kTargetRankOrder =
    "release_ack_send_i_if_required_then_data_recv_i_then_"
    "fill_ack_send_i_then_data_recv_i_plus_1";
constexpr const char* kSourceRankOrder =
    "release_ack_recv_i_if_required_then_data_send_i_then_fill_ack_recv_i";
constexpr const char* kTileBranching =
    "tile1_head_main_chain_tile2_immediate_side_branch_tile3_plus_release_gated";
constexpr const char* kShareRule =
    "quotient_remainder_earliest_global_indices";

[[noreturn]] void fail(const std::string& message) {
    throw LedgerError(message);
}

// ------------------------------------------------------------- arithmetic --

// _checked_add / _checked_mul: uint64 with overflow rejection.
u64 checked_add(u64 a, u64 b, const std::string& what) {
    if (a > U64_MAX - b) fail("uint64 overflow in " + what);
    return a + b;
}

u64 checked_mul(u64 a, u64 b, const std::string& what) {
    if (a != 0 && b > U64_MAX / a) fail("uint64 overflow in " + what);
    return a * b;
}

// _ceil_div(a, b) with b >= 1.
u64 ceil_div_u64(u64 a, u64 b) {
    return a / b + (a % b != 0 ? 1 : 0);
}

u128 ceil_div_u128(u128 a, u128 b) {
    return a / b + (a % b != 0 ? 1 : 0);
}

bool add_overflows_u128(u128 a, u128 b) {
    return a + b < a;  // wraps mod 2^128
}

bool mul_overflows_u128(u128 a, u128 b, u128* out) {
    if (a == 0 || b == 0) {
        *out = 0;
        return false;
    }
    const u128 product = a * b;
    if (product / a != b) return true;
    *out = product;
    return false;
}

// Python floor division / modulo for positive divisors over signed 128-bit.
i128 py_div(i128 a, i128 b) {  // b > 0
    i128 q = a / b;
    const i128 r = a % b;
    if (r != 0 && r < 0) --q;
    return q;
}

i128 py_mod(i128 a, i128 b) {  // b > 0, result in [0, b)
    i128 r = a % b;
    if (r < 0) r += b;
    return r;
}

// pow(x, -1, m) for gcd(x, m) == 1, m >= 1 (result in [0, m)).
u64 mod_inverse(u64 a, u64 m) {
    if (m == 1) return 0;
    i128 t = 0;
    i128 new_t = 1;
    u64 r = m;
    u64 new_r = a % m;
    while (new_r != 0) {
        const u64 q = r / new_r;
        const i128 next_t = t - static_cast<i128>(q) * new_t;
        t = new_t;
        new_t = next_t;
        const u64 next_r = r - q * new_r;
        r = new_r;
        new_r = next_r;
    }
    return static_cast<u64>(py_mod(t, static_cast<i128>(m)));
}

// --------------------------------------------------------------- readers --

// Local variants of the Python scalar validators (_uint default maximum is
// U64_MAX, _positive requires > 0, _int64 allows negatives); the shared
// LedgerTypes readers provide the exact type lattice.
u64 p_uint(const Json& value, const std::string& path, u64 maximum = U64_MAX) {
    return read_uint(value, path, maximum);
}

u64 p_positive(const Json& value, const std::string& path,
               u64 maximum = U64_MAX) {
    const u64 result = read_uint(value, path, maximum);
    if (result == 0) fail(path + " must be positive");
    return result;
}

i64 p_int64(const Json& value, const std::string& path) {
    if (value.is_boolean() || !value.is_number_integer()) {
        fail(path + " must be a signed 64-bit integer (bool is invalid)");
    }
    if (value.is_number_unsigned()) {
        const u64 raw = value.get<u64>();
        if (raw > static_cast<u64>(kI64Max)) {
            fail(path + " must be a signed 64-bit integer (bool is invalid)");
        }
        return static_cast<i64>(raw);
    }
    return value.get<i64>();
}

u64 p_rank(const Json& value, const std::string& path) {
    return read_rank(value, path);
}

std::string p_string(const Json& value, const std::string& path) {
    return read_string(value, path);
}

std::string p_enum(const Json& value, const std::string& path,
                   std::initializer_list<const char*> allowed) {
    if (value.is_string()) {
        const std::string text = value.get<std::string>();
        for (const char* candidate : allowed) {
            if (text == candidate) return text;
        }
    }
    fail(path + " must be one of the allowed values, got " + value.dump());
}

// _mapping: object with exact required key set and known optional keys.
const Json& mapping(const Json& value, const std::string& path,
                    const std::vector<const char*>& required,
                    const std::vector<const char*>& optional = {}) {
    if (!value.is_object()) fail(path + " must be an object");
    std::vector<const char*> allowed(required);
    allowed.insert(allowed.end(), optional.begin(), optional.end());
    require_keys(value, path, allowed, required);
    return value;
}

const Json& p_array(const Json& value, const std::string& path,
                    bool nonempty = false) {
    if (!value.is_array()) fail(path + " must be an array");
    if (nonempty && value.empty()) fail(path + " must not be empty");
    return value;
}

std::vector<std::string> string_array(const Json& value, const std::string& path) {
    const Json& items = p_array(value, path);
    std::vector<std::string> out;
    out.reserve(items.size());
    for (std::size_t i = 0; i < items.size(); ++i) {
        out.push_back(read_string(items[i], path + "[" + std::to_string(i) + "]"));
    }
    return out;
}

std::string at_index(const std::string& path, std::size_t i) {
    return path + "[" + std::to_string(i) + "]";
}

std::string at_field(const std::string& path, const char* field) {
    return path + "." + field;
}

// str.encode("utf-8").hex() over the parsed JSON string bytes.
std::string hex_encode(const std::string& text) {
    static const char* kDigits = "0123456789abcdef";
    std::string out;
    out.reserve(text.size() * 2);
    for (char raw : text) {
        const unsigned char byte = static_cast<unsigned char>(raw);
        out.push_back(kDigits[byte >> 4]);
        out.push_back(kDigits[byte & 0xF]);
    }
    return out;
}

// ------------------------------------------------------ local parse types --

// Sections the header keeps as canonical Json only; typed locals used during
// from_wire validation.
struct IdRange {
    u64 rank = 0;
    u64 first = 0;
    u64 count = 0;
    u64 stride = 0;
};

struct TagRun {
    std::string allocator_scope_id;
    u64 emission_ordinal = 0;
    u64 first_tag = 0;
    u64 count = 0;
    u64 group_ordinal = 0;
    u64 first_tile = 0;
    u64 first_shard = 0;
    u64 tile_count = 0;
    u64 shard_count = 0;
};

// Shared shape of endpoint_dependency_runs and consume_completion_runs rows.
struct DependencyRun {
    u64 rank = 0;
    u64 group_ordinal = 0;
    u64 first_tile = 0;
    u64 tile_count = 0;
    std::vector<u64> shard_ordinals;
    bool has_endpoint = false;
    std::string endpoint;
    std::string dependency_kind;
    u64 first_dependency_json_id = 0;
    u64 dependency_tile_stride = 0;
    u64 dependency_shard_stride = 0;
};

struct ProjectionShare {
    std::string rule;
    u64 total_units = 0;
    u64 total_tiles = 0;
    u64 share_index_first = 0;
    u64 tile_bytes = 0;
};

struct ProjectionFill {
    std::string leg_id_prefix;
    u64 first_index = 0;
    u64 fill_order_first = 0;
    u64 fill_order_stride = 0;
    std::optional<i64> completion_order_first;
    i64 completion_order_stride = 0;
    std::vector<std::string> route;
    u64 bytes_per_tile = 0;
    u64 latency_ns = 0;
    std::vector<std::string> start_deps;
    std::string gate_id;
    u64 gate_capacity = 0;
};

struct ProjectionConsume {
    std::string leg_id_prefix;
    u64 first_index = 0;
    u64 consume_order_first = 0;
    u64 consume_order_stride = 0;
    std::optional<i64> completion_order_first;
    i64 completion_order_stride = 0;
    std::string kind;
    std::vector<std::string> route;
    std::optional<std::string> group;
    u64 latency_first_ns = 0;
    u64 latency_rest_ns = 0;
    std::vector<std::string> consume_deps;
    std::vector<std::string> consume_start_deps;
    std::string gate_id;
    std::optional<ProjectionShare> compute_share;
    std::optional<ProjectionShare> hbm_share;
};

struct ProjectionChain {
    u64 group_ordinal = 0;
    u64 rank = 0;
    u64 tile_count = 0;
    ProjectionFill fill;
    ProjectionConsume consume;
};

struct AnchorSelector {
    std::string kind;
    std::optional<u64> group_ordinal;
    std::optional<u64> tile_ordinal;
    std::optional<u64> shard_ordinal;
    std::optional<std::string> endpoint;
    std::optional<std::string> existing_gate;
    std::optional<u64> tag_allocation_emission_ordinal;
    std::optional<u64> tag_offset;

    bool operator==(const AnchorSelector& other) const {
        return kind == other.kind && group_ordinal == other.group_ordinal
            && tile_ordinal == other.tile_ordinal
            && shard_ordinal == other.shard_ordinal
            && endpoint == other.endpoint
            && existing_gate == other.existing_gate
            && tag_allocation_emission_ordinal
                == other.tag_allocation_emission_ordinal
            && tag_offset == other.tag_offset;
    }
};

struct Anchor {
    std::string role;
    u64 rank = 0;
    std::string binding;
    std::optional<u64> node_id;
    AnchorSelector selector;
};

// _ENDPOINTS tuple membership is enforced by the enum validators below.
u64 endpoint_owner_rank(const PipelineShardRule& shard,
                        const std::string& endpoint) {
    const bool source_side = endpoint == "data_send"
        || endpoint == "fill_ack_recv" || endpoint == "release_ack_recv";
    return source_side ? shard.source_rank : shard.target_rank;
}

// ----------------------------------------------------- progression algebra --

// (rank, first, stride, count) arithmetic progression; Python evaluates
// first + stride * (count - 1) in the unbounded integer domain, so the last
// ID may exceed uint64 and every helper below works in 128-bit.
struct Progression {
    u64 rank;
    u64 first;
    u64 stride;
    u64 count;
};

u128 progression_last(const Progression& p) {
    return static_cast<u128>(p.first)
        + static_cast<u128>(p.stride) * (p.count - 1);
}

// Smallest common ID of two progressions that is >= max(firsts); -1 when the
// progressions share no ID inside [low, high].  Exact port of the CRT logic
// shared by _progressions_intersect / _progression_intersection_count.
i128 first_common_id(const Progression& left, const Progression& right) {
    if (left.rank != right.rank) return -1;
    if (left.count == 0 || right.count == 0) return -1;
    const u128 left_last = progression_last(left);
    const u128 right_last = progression_last(right);
    const u128 low = std::max<u128>(left.first, right.first);
    const u128 high = std::min(left_last, right_last);
    if (low > high) return -1;
    const u64 common = std::gcd(left.stride, right.stride);
    const i128 delta = static_cast<i128>(right.first)
        - static_cast<i128>(left.first);
    if (py_mod(delta, static_cast<i128>(common)) != 0) return -1;
    const u64 modulus = right.stride / common;
    u128 step = 0;
    if (modulus != 1) {
        const u64 base = static_cast<u64>(py_mod(
            py_div(delta, static_cast<i128>(common)),
            static_cast<i128>(modulus)));
        const u64 inverse = mod_inverse((left.stride / common) % modulus, modulus);
        step = (static_cast<u128>(base) * inverse) % modulus;
    }
    u128 first = static_cast<u128>(left.first)
        + static_cast<u128>(left.stride) * step;
    const u128 period = static_cast<u128>(left.stride / common)
        * static_cast<u128>(right.stride);
    if (first < low) {
        const u128 k = ceil_div_u128(low - first, period);
        u128 bump = 0;
        if (mul_overflows_u128(k, period, &bump)
                || add_overflows_u128(first, bump)) {
            return -1;  // next candidate >= 2^128, hence beyond high
        }
        first += bump;
    }
    if (first > high) return -1;
    return static_cast<i128>(first);
}

// _progressions_intersect.
bool progressions_intersect(const Progression& left, const Progression& right) {
    return first_common_id(left, right) >= 0;
}

// _progression_intersection_count (exact common-ID count).
u128 progression_intersection_count(const Progression& left,
                                    const Progression& right) {
    const i128 hit = first_common_id(left, right);
    if (hit < 0) return 0;
    const u64 common = std::gcd(left.stride, right.stride);
    const u128 period = static_cast<u128>(left.stride / common)
        * static_cast<u128>(right.stride);
    const u128 high = std::min(progression_last(left), progression_last(right));
    return (high - static_cast<u128>(hit)) / period + 1;
}

// _EndpointIdIndex: per-rank hole/anchor index for endpoint-ID coverage.
class EndpointIdIndex {
  public:
    struct HoleEntry {
        u64 first;
        u64 last;
        Progression hole;
    };

    EndpointIdIndex(const std::vector<IdRange>& holes,
                    const std::vector<Anchor>& anchors) {
        std::map<u64, std::vector<HoleEntry>> by_rank;
        for (const IdRange& hole : holes) {
            const u128 last = static_cast<u128>(hole.first)
                + static_cast<u128>(hole.stride) * (hole.count - 1);
            by_rank[hole.rank].push_back(HoleEntry{
                hole.first, static_cast<u64>(last),
                Progression{hole.rank, hole.first, hole.stride, hole.count}});
        }
        for (auto& entry : by_rank) {
            std::vector<HoleEntry>& items = entry.second;
            std::stable_sort(items.begin(), items.end(),
                             [](const HoleEntry& a, const HoleEntry& b) {
                                 return a.first < b.first;
                             });
            std::vector<u64> firsts;
            std::vector<u64> prefix_max;
            firsts.reserve(items.size());
            bool have_running = false;
            u64 running = 0;
            for (const HoleEntry& item : items) {
                firsts.push_back(item.first);
                running = have_running ? std::max(running, item.last) : item.last;
                have_running = true;
                prefix_max.push_back(running);
            }
            holes_by_rank_[entry.first] = std::move(items);
            hole_firsts_[entry.first] = std::move(firsts);
            hole_prefix_max_[entry.first] = std::move(prefix_max);
        }
        for (const Anchor& anchor : anchors) {
            if (anchor.binding != "physical_endpoint_slot" || !anchor.node_id) {
                continue;
            }
            anchors_by_rank_[anchor.rank].push_back(*anchor.node_id);
        }
        for (auto& entry : anchors_by_rank_) {
            std::sort(entry.second.begin(), entry.second.end());
        }
    }

    // covered_count(rank, first, stride, count).
    u128 covered_count(u64 rank, u64 first, u64 stride, u64 count) const {
        u128 total = 0;
        const u128 last = static_cast<u128>(first)
            + static_cast<u128>(stride) * (count - 1);
        auto holes = holes_by_rank_.find(rank);
        if (holes != holes_by_rank_.end()) {
            const std::vector<HoleEntry>& items = holes->second;
            const std::vector<u64>& firsts = hole_firsts_.at(rank);
            const std::vector<u64>& prefix_max = hole_prefix_max_.at(rank);
            const std::size_t stop = bisect_right(firsts, static_cast<i128>(last));
            const Progression progression{rank, first, stride, count};
            for (std::size_t index = stop; index-- > 0;) {
                if (prefix_max[index] < first) break;
                const HoleEntry& entry = items[index];
                if (entry.last < first || entry.first > last) continue;
                total += progression_intersection_count(progression, entry.hole);
            }
        }
        auto anchors = anchors_by_rank_.find(rank);
        if (anchors != anchors_by_rank_.end()) {
            const std::vector<u64>& ids = anchors->second;
            const std::size_t lo = bisect_right(
                ids, static_cast<i128>(first) - 1);
            const std::size_t hi = bisect_right(ids, static_cast<i128>(last));
            for (std::size_t index = lo; index < hi; ++index) {
                const u128 delta = static_cast<u128>(ids[index])
                    - static_cast<u128>(first);
                if (delta % stride == 0) total += 1;
            }
        }
        return total;
    }

    // Exact singleton-vs-hole membership.
    bool anchor_hits_hole(u64 rank, u64 node_id) const {
        auto holes = holes_by_rank_.find(rank);
        if (holes == holes_by_rank_.end()) return false;
        const std::vector<HoleEntry>& items = holes->second;
        const std::vector<u64>& firsts = hole_firsts_.at(rank);
        const std::vector<u64>& prefix_max = hole_prefix_max_.at(rank);
        const std::size_t stop = bisect_right(firsts, static_cast<i128>(node_id));
        for (std::size_t index = stop; index-- > 0;) {
            const HoleEntry& entry = items[index];
            if (entry.last < node_id) {
                if (prefix_max[index] < node_id) break;
                continue;
            }
            if (node_id >= entry.first
                    && (node_id - entry.first) % entry.hole.stride == 0) {
                return true;
            }
        }
        return false;
    }

  private:
    // bisect_right(values, bound) for possibly out-of-uint64 bounds.
    static std::size_t bisect_right(const std::vector<u64>& values, i128 bound) {
        if (bound < 0) return 0;
        if (bound > static_cast<i128>(U64_MAX)) return values.size();
        return static_cast<std::size_t>(
            std::upper_bound(values.begin(), values.end(),
                             static_cast<u64>(bound)) - values.begin());
    }

    std::map<u64, std::vector<HoleEntry>> holes_by_rank_;
    std::map<u64, std::vector<u64>> hole_firsts_;
    std::map<u64, std::vector<u64>> hole_prefix_max_;
    std::map<u64, std::vector<u64>> anchors_by_rank_;
};

// ------------------------------------------------------------ group parse --

// _parse_group.
PipelineGroup parse_group(const Json& value, const std::string& path) {
    const Json& raw = mapping(value, path,
        {"group_ordinal", "pass_or_window_ordinal", "source_ordinal",
         "source_key", "layer_start", "layer_end", "tile_axis", "shards"});
    const Json& axis_raw = mapping(raw["tile_axis"], path + ".tile_axis",
        {"unit", "total_units", "units_per_tile", "tile_count", "tail_units",
         "global_tile_first"});
    const std::string unit = p_enum(axis_raw["unit"], path + ".tile_axis.unit",
                                    {"source_tokens", "stream_bytes"});
    const u64 total_units = p_positive(axis_raw["total_units"],
                                       path + ".tile_axis.total_units");
    const u64 units_per_tile = p_positive(axis_raw["units_per_tile"],
                                          path + ".tile_axis.units_per_tile");
    const u64 tile_count = p_positive(axis_raw["tile_count"],
                                      path + ".tile_axis.tile_count");
    const u64 tail_units = p_uint(axis_raw["tail_units"],
                                  path + ".tile_axis.tail_units");
    const u64 global_first = p_positive(axis_raw["global_tile_first"],
                                        path + ".tile_axis.global_tile_first");
    const u64 expected_count = ceil_div_u64(total_units, units_per_tile);
    const u64 expected_tail =
        total_units - (expected_count - 1) * units_per_tile;
    if (tile_count != expected_count || tail_units != expected_tail) {
        fail(path + ".tile_axis has inconsistent count/tail");
    }
    checked_add(global_first, tile_count - 1, "global tile range");

    PipelineGroup group;
    group.tile_axis.unit = unit;
    group.tile_axis.total_units = total_units;
    group.tile_axis.units_per_tile = units_per_tile;
    group.tile_axis.tile_count = tile_count;
    group.tile_axis.tail_units = tail_units;
    group.tile_axis.global_tile_first = global_first;

    const Json& shards_raw = p_array(raw["shards"], path + ".shards", true);
    group.shards.reserve(shards_raw.size());
    for (std::size_t i = 0; i < shards_raw.size(); ++i) {
        const std::string shard_path = at_index(path + ".shards", i);
        const Json& shard = mapping(shards_raw[i], shard_path,
            {"shard_ordinal", "source_rank", "target_rank",
             "total_data_bytes", "bytes_per_unit", "noc_path"});
        const u64 ordinal = p_uint(shard["shard_ordinal"],
                                   at_field(shard_path, "shard_ordinal"));
        if (ordinal != i) {
            fail(shard_path + ".shard_ordinal must be original zero-based order");
        }
        const u64 source_rank = p_rank(shard["source_rank"],
                                       at_field(shard_path, "source_rank"));
        const u64 target_rank = p_rank(shard["target_rank"],
                                       at_field(shard_path, "target_rank"));
        const u64 total_bytes = p_uint(shard["total_data_bytes"],
                                       at_field(shard_path, "total_data_bytes"));
        const u64 bytes_per_unit = p_uint(shard["bytes_per_unit"],
                                          at_field(shard_path, "bytes_per_unit"));
        std::vector<u64> noc_path;
        const Json& path_raw = p_array(shard["noc_path"],
                                       at_field(shard_path, "noc_path"));
        noc_path.reserve(path_raw.size());
        for (std::size_t j = 0; j < path_raw.size(); ++j) {
            noc_path.push_back(p_rank(
                path_raw[j],
                at_field(shard_path, "noc_path") + "[" + std::to_string(j) + "]"));
        }
        if (!noc_path.empty()
                && (noc_path.front() != source_rank
                    || noc_path.back() != target_rank)) {
            fail(shard_path + ".noc_path must begin/end at shard ranks");
        }
        if (unit == "source_tokens") {
            if (bytes_per_unit == 0
                    || checked_mul(total_units, bytes_per_unit,
                                   "shard total bytes") != total_bytes) {
                fail(shard_path
                     + ".total_data_bytes must equal total_units*bytes_per_unit");
            }
        } else if (bytes_per_unit != 1 || total_bytes > total_units) {
            fail(shard_path + " has invalid stream_bytes extent/bytes_per_unit");
        }
        group.shards.push_back(
            PipelineShardRule{ordinal, source_rank, target_rank, total_bytes,
                              bytes_per_unit, std::move(noc_path)});
    }
    if (unit == "stream_bytes") {
        u64 max_bytes = 0;
        for (const PipelineShardRule& shard : group.shards) {
            max_bytes = std::max(max_bytes, shard.total_data_bytes);
        }
        if (max_bytes != total_units) {
            fail(path + ".tile_axis.total_units must equal max shard bytes");
        }
    }
    const u64 layer_start = p_uint(raw["layer_start"], path + ".layer_start");
    const u64 layer_end = p_positive(raw["layer_end"], path + ".layer_end");
    if (layer_end <= layer_start) {
        fail(path + " layer range must be nonempty");
    }
    group.group_ordinal = p_uint(raw["group_ordinal"], path + ".group_ordinal");
    group.pass_or_window_ordinal = p_uint(raw["pass_or_window_ordinal"],
                                          path + ".pass_or_window_ordinal");
    group.source_ordinal = p_uint(raw["source_ordinal"],
                                  path + ".source_ordinal");
    group.source_key = p_string(raw["source_key"], path + ".source_key");
    group.layer_start = layer_start;
    group.layer_end = layer_end;
    return group;
}

// --------------------------------------------------------- protocol parse --

// _parse_protocol: frozen two-slot lag-2 contract.
PipelineProtocol parse_protocol(const Json& value) {
    const Json& raw = mapping(value, "protocol",
        {"buffer_slots", "release_lag_tiles", "initial_tiles_without_release",
         "fill_ack_bytes", "release_ack_bytes", "hbm_charge",
         "target_rank_order", "source_rank_order", "tile_branching"});
    PipelineProtocol protocol;
    protocol.buffer_slots = p_positive(raw["buffer_slots"],
                                       "protocol.buffer_slots", kU32Max);
    protocol.release_lag_tiles = p_positive(raw["release_lag_tiles"],
                                            "protocol.release_lag_tiles",
                                            kU32Max);
    protocol.initial_tiles_without_release = p_positive(
        raw["initial_tiles_without_release"],
        "protocol.initial_tiles_without_release", kU32Max);
    protocol.fill_ack_bytes = p_positive(raw["fill_ack_bytes"],
                                         "protocol.fill_ack_bytes");
    protocol.release_ack_bytes = p_positive(raw["release_ack_bytes"],
                                            "protocol.release_ack_bytes");
    protocol.hbm_charge = read_bool(raw["hbm_charge"], "protocol.hbm_charge");
    protocol.target_rank_order = p_string(raw["target_rank_order"],
                                          "protocol.target_rank_order");
    protocol.source_rank_order = p_string(raw["source_rank_order"],
                                          "protocol.source_rank_order");
    protocol.tile_branching = p_string(raw["tile_branching"],
                                       "protocol.tile_branching");
    if (!protocol.hbm_charge
            || protocol.buffer_slots != 2
            || protocol.release_lag_tiles != 2
            || protocol.initial_tiles_without_release != 2
            || protocol.fill_ack_bytes != 1
            || protocol.release_ack_bytes != 1
            || protocol.target_rank_order != kTargetRankOrder
            || protocol.source_rank_order != kSourceRankOrder
            || protocol.tile_branching != kTileBranching) {
        fail("protocol must match the frozen two-slot lag-2 contract");
    }
    return protocol;
}

// ---------------------------------------------- prediction/execution parse --

// _parse_prediction_profile; returns the canonical to_wire section.
Json parse_prediction_profile(const Json& value) {
    const Json& raw = mapping(value, "prediction_profile",
                              {"prefill_consume", "decode_consume"});
    if (!(raw["prefill_consume"] == Json("zero_duration_no_extra_consume"))) {
        fail("prediction_profile.prefill_consume is fixed");
    }
    const Json& decode = mapping(raw["decode_consume"],
        "prediction_profile.decode_consume",
        {"kind", "profile_id", "compute_ns_total", "hbm_bytes_total",
         "resource_key", "share_rule"});
    const double compute = read_finite_nonnegative(
        decode["compute_ns_total"], "decode_consume.compute_ns_total");
    const std::string kind = p_enum(decode["kind"], "decode_consume.kind",
                                    {"profiled_consume"});
    const std::string profile_id = p_string(decode["profile_id"],
                                            "decode_consume.profile_id");
    const u64 hbm_bytes = p_uint(decode["hbm_bytes_total"],
                                 "decode_consume.hbm_bytes_total");
    const std::string resource_key = p_string(decode["resource_key"],
                                              "decode_consume.resource_key");
    const std::string share_rule = p_enum(
        decode["share_rule"], "decode_consume.share_rule",
        {"existing_action_work_graph_byte_share_vector"});
    Json decode_json;
    decode_json["kind"] = kind;
    decode_json["profile_id"] = profile_id;
    decode_json["compute_ns_total"] = compute;  // float(compute) like Python
    decode_json["hbm_bytes_total"] = hbm_bytes;
    decode_json["resource_key"] = resource_key;
    decode_json["share_rule"] = share_rule;
    Json out;
    out["prefill_consume"] = "zero_duration_no_extra_consume";
    out["decode_consume"] = std::move(decode_json);
    return out;
}

// _parse_execution_profile; returns the canonical to_wire section.
Json parse_execution_profile(const Json& value) {
    static const char* kKeys[4] = {"data_legs", "endpoint_hbm_rule",
                                   "prefill_consume", "decode_consume"};
    static const char* kExpected[4] = {
        "ordinary_comm_send_recv_path",
        "one_original_COMM_READ_or_COMM_WRITE_job_per_charged_endpoint",
        "existing_body_or_marker_only", "existing_body_nodes_only"};
    const Json& raw = mapping(value, "execution_profile",
                              {kKeys[0], kKeys[1], kKeys[2], kKeys[3]});
    Json out;
    for (int i = 0; i < 4; ++i) {
        const std::string text =
            p_string(raw[kKeys[i]], std::string("execution_profile.") + kKeys[i]);
        if (text != kExpected[i]) {
            fail("execution_profile must preserve original COMM/HBM/body semantics");
        }
        out[kKeys[i]] = text;
    }
    return out;
}

// --------------------------------------------------------- id range parse --

// _parse_id_range.
IdRange parse_id_range(const Json& value, const std::string& path) {
    const Json& raw = mapping(value, path,
        {"rank", "first_json_id", "count", "stride", "kind"});
    IdRange out;
    out.rank = p_rank(raw["rank"], path + ".rank");
    out.first = p_uint(raw["first_json_id"], path + ".first_json_id");
    out.count = p_positive(raw["count"], path + ".count");
    out.stride = p_positive(raw["stride"], path + ".stride");
    // Unbounded-domain terminal-ID check (uint128 intermediate in Python).
    const u128 last = static_cast<u128>(out.first)
        + static_cast<u128>(out.stride) * (out.count - 1);
    if (last > static_cast<u128>(U64_MAX)) {
        fail(path + " terminal ID exceeds uint64");
    }
    p_enum(raw["kind"], path + ".kind", {"virtual_endpoint_hole"});
    return out;
}

// _validate_id_ranges.
void validate_id_ranges(const std::vector<IdRange>& ranges) {
    i64 last_rank = -1;
    std::map<u64, u64> last_first;
    std::map<u64, std::vector<std::pair<u64, Progression>>> active_by_rank;
    for (const IdRange& item : ranges) {
        if (static_cast<i64>(item.rank) < last_rank) {
            fail("logical ID ranges must be rank ordered");
        }
        auto previous_first = last_first.find(item.rank);
        if (previous_first != last_first.end()
                && item.first < previous_first->second) {
            fail("logical ID ranges must be ordered by first_json_id per rank");
        }
        const Progression current{item.rank, item.first, item.stride, item.count};
        const u64 last_id = static_cast<u64>(
            static_cast<u128>(item.first)
            + static_cast<u128>(item.stride) * (item.count - 1));
        std::vector<std::pair<u64, Progression>>& actives = active_by_rank[item.rank];
        std::vector<std::pair<u64, Progression>> kept;
        kept.reserve(actives.size() + 1);
        for (const auto& prior : actives) {
            if (prior.first < item.first) continue;  // retired below current start
            if (progressions_intersect(current, prior.second)) {
                fail("logical ID arithmetic progressions overlap");
            }
            kept.push_back(prior);
        }
        kept.emplace_back(last_id, current);
        actives = std::move(kept);
        last_first[item.rank] = item.first;
        last_rank = static_cast<i64>(item.rank);
    }
}

// ------------------------------------------------------------ tag runs --

// _parse_tag_run.
TagRun parse_tag_run(const Json& value, const std::string& path) {
    const Json& raw = mapping(value, path,
        {"allocator_scope_id", "emission_ordinal", "first_tag", "count",
         "group_ordinal", "first_tile", "first_shard", "tile_count",
         "shard_count"});
    TagRun out;
    out.allocator_scope_id = p_string(raw["allocator_scope_id"],
                                      at_field(path, "allocator_scope_id"));
    out.emission_ordinal = p_uint(raw["emission_ordinal"],
                                  at_field(path, "emission_ordinal"));
    out.first_tag = p_positive(raw["first_tag"], at_field(path, "first_tag"),
                               kU32Max);
    out.count = p_positive(raw["count"], at_field(path, "count"), kU32Max);
    out.group_ordinal = p_uint(raw["group_ordinal"],
                               at_field(path, "group_ordinal"));
    out.first_tile = p_positive(raw["first_tile"], at_field(path, "first_tile"));
    out.first_shard = p_uint(raw["first_shard"], at_field(path, "first_shard"));
    out.tile_count = p_positive(raw["tile_count"], at_field(path, "tile_count"));
    out.shard_count = p_positive(raw["shard_count"],
                                 at_field(path, "shard_count"));
    return out;
}

std::map<u64, const PipelineGroup*> groups_by_ordinal(
        const std::vector<PipelineGroup>& groups) {
    std::map<u64, const PipelineGroup*> out;
    for (const PipelineGroup& group : groups) {
        out[group.group_ordinal] = &group;
    }
    return out;
}

// _validate_tag_runs.
void validate_tag_runs(const std::vector<TagRun>& runs,
                       const std::vector<PipelineGroup>& groups) {
    const std::map<u64, const PipelineGroup*> by_id = groups_by_ordinal(groups);
    std::map<std::string, std::pair<u64, u64>> last;
    for (const TagRun& run : runs) {
        auto group_it = by_id.find(run.group_ordinal);
        if (group_it == by_id.end()) {
            fail("tag run references unknown group");
        }
        const PipelineGroup& group = *group_it->second;
        if (static_cast<i128>(run.first_tile) + run.tile_count - 1
                > static_cast<i128>(group.tile_axis.tile_count)) {
            fail("tag run tile range exceeds group");
        }
        if (static_cast<i128>(run.first_shard) + run.shard_count
                > static_cast<i128>(group.shards.size())) {
            fail("tag run shard range exceeds group");
        }
        const u64 tag_end = run.first_tag + run.count;  // <= 2^32, no overflow
        if (tag_end > kU32Max + 1) {
            fail("tag run exceeds uint32 tag space");
        }
        auto previous = last.find(run.allocator_scope_id);
        if (previous != last.end()
                && (run.emission_ordinal <= previous->second.first
                    || run.first_tag < previous->second.second)) {
            fail("tag runs must preserve allocator emission/tag order");
        }
        last[run.allocator_scope_id] = {run.emission_ordinal, tag_end};
    }
}

// ------------------------------------------------------- endpoint runs --

// _parse_endpoint_run.
PipelineEndpointRun parse_endpoint_run(const Json& value,
                                       const std::string& path) {
    const Json& raw = mapping(value, path,
        {"rank", "group_ordinal", "first_tile", "tile_count",
         "shard_ordinals", "first_json_id", "tile_stride", "shard_stride",
         "initial_parent_json_id", "legacy_context", "legs"});
    std::vector<u64> shards;
    const Json& shards_raw = p_array(raw["shard_ordinals"],
                                     path + ".shard_ordinals", true);
    shards.reserve(shards_raw.size());
    for (std::size_t i = 0; i < shards_raw.size(); ++i) {
        shards.push_back(p_uint(
            shards_raw[i], path + ".shard_ordinals[" + std::to_string(i) + "]"));
    }
    for (std::size_t i = 1; i < shards.size(); ++i) {
        if (shards[i - 1] >= shards[i]) {
            fail(path + ".shard_ordinals must be strictly increasing");
        }
    }
    const std::optional<u64> parent = read_nullable_uint(
        raw["initial_parent_json_id"], path + ".initial_parent_json_id");
    const Json& context_raw = mapping(raw["legacy_context"],
                                      path + ".legacy_context",
                                      {"request_id", "stage", "generation"});
    const std::string context_stage =
        p_enum(context_raw["stage"], path + ".legacy_context.stage",
               {"prefill", "decode", "completion"});
    const u64 context_generation =
        p_uint(context_raw["generation"], path + ".legacy_context.generation");
    if (context_generation != (context_stage == "prefill" ? 0u : 1u)) {
        fail(path + ".legacy_context generation disagrees with stage");
    }
    PipelineEndpointRun run;
    run.legacy_context.stage = context_stage;
    run.legacy_context.generation = context_generation;
    run.legacy_context.request_id = p_string(
        context_raw["request_id"], path + ".legacy_context.request_id");

    const Json& legs_raw = p_array(raw["legs"], path + ".legs", true);
    run.legs.reserve(legs_raw.size());
    for (std::size_t i = 0; i < legs_raw.size(); ++i) {
        const std::string leg_path = at_index(path + ".legs", i);
        const Json& leg = mapping(legs_raw[i], leg_path,
            {"endpoint", "json_offset", "tag_emission_ordinal", "tag_offset",
             "tag_tile_stride", "tag_shard_stride"});
        const std::string endpoint = p_enum(
            leg["endpoint"], leg_path + ".endpoint",
            {"data_send", "data_recv", "fill_ack_send", "fill_ack_recv",
             "release_ack_send", "release_ack_recv", "body_marker"});
        const u64 json_offset =
            p_uint(leg["json_offset"], leg_path + ".json_offset");
        std::optional<u64> emission;
        std::optional<u64> tag_offset;
        std::optional<u64> tag_tile_stride;
        std::optional<u64> tag_shard_stride;
        if (endpoint == "body_marker") {
            if (!(leg["tag_emission_ordinal"].is_null()
                    && leg["tag_offset"].is_null()
                    && leg["tag_tile_stride"].is_null()
                    && leg["tag_shard_stride"].is_null())) {
                fail(leg_path + " body_marker tag fields must be null");
            }
        } else {
            if (leg["tag_emission_ordinal"].is_null()
                    || leg["tag_offset"].is_null()
                    || leg["tag_tile_stride"].is_null()
                    || leg["tag_shard_stride"].is_null()) {
                fail(leg_path + " requires all tag fields");
            }
            emission = p_uint(leg["tag_emission_ordinal"],
                              leg_path + ".tag_emission_ordinal");
            tag_offset = p_uint(leg["tag_offset"], leg_path + ".tag_offset",
                                kU32Max);
            tag_tile_stride = p_positive(leg["tag_tile_stride"],
                                         leg_path + ".tag_tile_stride");
            tag_shard_stride = p_positive(leg["tag_shard_stride"],
                                          leg_path + ".tag_shard_stride");
        }
        run.legs.push_back(PipelineEndpointLegRun{
            endpoint, json_offset, emission, tag_offset, tag_tile_stride,
            tag_shard_stride});
    }
    for (std::size_t i = 0; i < run.legs.size(); ++i) {
        for (std::size_t j = i + 1; j < run.legs.size(); ++j) {
            if (run.legs[i].endpoint == run.legs[j].endpoint) {
                fail(path + ".legs endpoint kinds must be unique");
            }
            if (run.legs[i].json_offset == run.legs[j].json_offset) {
                fail(path + ".legs json_offset values must be unique");
            }
        }
    }
    run.rank = p_rank(raw["rank"], path + ".rank");
    run.group_ordinal = p_uint(raw["group_ordinal"], path + ".group_ordinal");
    run.first_tile = p_positive(raw["first_tile"], path + ".first_tile");
    run.tile_count = p_positive(raw["tile_count"], path + ".tile_count");
    run.shard_ordinals = std::move(shards);
    run.first_json_id = p_uint(raw["first_json_id"], path + ".first_json_id");
    run.tile_stride = p_positive(raw["tile_stride"], path + ".tile_stride");
    run.shard_stride = p_positive(raw["shard_stride"], path + ".shard_stride");
    run.initial_parent_json_id = parent;
    return run;
}

// _validate_endpoint_runs.
void validate_endpoint_runs(
        const std::vector<PipelineEndpointRun>& endpoint_runs,
        const std::vector<PipelineGroup>& groups,
        const std::vector<TagRun>& tag_runs) {
    const std::map<u64, const PipelineGroup*> group_by_id =
        groups_by_ordinal(groups);
    std::map<u64, const TagRun*> tag_by_emission;
    for (const TagRun& tag_run : tag_runs) {
        if (tag_by_emission.count(tag_run.emission_ordinal)) {
            fail("tag emission ordinals must be globally unique");
        }
        tag_by_emission[tag_run.emission_ordinal] = &tag_run;
    }
    for (const PipelineEndpointRun& run : endpoint_runs) {
        auto group_it = group_by_id.find(run.group_ordinal);
        if (group_it == group_by_id.end()) {
            fail("endpoint run references unknown group");
        }
        const PipelineGroup& group = *group_it->second;
        const u64 tile_end = checked_add(run.first_tile, run.tile_count - 1,
                                         "endpoint tile range");
        if (tile_end > group.tile_axis.tile_count) {
            fail("endpoint run tile range exceeds group");
        }
        for (u64 shard : run.shard_ordinals) {
            if (shard >= group.shards.size()) {
                fail("endpoint run shard is outside group");
            }
        }
        const PipelineTileAxis& axis = group.tile_axis;
        bool has_comm_leg = false;
        for (const PipelineEndpointLegRun& leg : run.legs) {
            if (leg.endpoint != "body_marker") {
                has_comm_leg = true;
                break;
            }
        }
        for (u64 shard_index : run.shard_ordinals) {
            const PipelineShardRule& shard =
                group.shards[static_cast<std::size_t>(shard_index)];
            const u64 active_count =
                axis.unit == "source_tokens"
                    ? axis.tile_count
                    : std::min(axis.tile_count,
                               ceil_div_u64(shard.total_data_bytes,
                                            axis.units_per_tile));
            if (has_comm_leg && tile_end > active_count) {
                fail("COMM endpoint run exceeds this shard's active tile cap");
            }
        }
        for (const PipelineEndpointLegRun& leg : run.legs) {
            if (leg.endpoint == "body_marker") continue;
            for (u64 shard_index : run.shard_ordinals) {
                const u64 owner = endpoint_owner_rank(
                    group.shards[static_cast<std::size_t>(shard_index)],
                    leg.endpoint);
                if (owner != run.rank) {
                    fail("endpoint run rank disagrees with shard route");
                }
            }
        }
        const u64 max_shard_local = run.shard_ordinals.size() - 1;
        u64 max_leg_offset = 0;
        for (const PipelineEndpointLegRun& leg : run.legs) {
            max_leg_offset = std::max(max_leg_offset, leg.json_offset);
        }
        if (run.shard_stride <= max_leg_offset) {
            fail("endpoint shard_stride overlaps leg IDs");
        }
        const u64 max_within_tile = checked_add(
            checked_mul(max_shard_local, run.shard_stride,
                        "endpoint shard ID extent"),
            max_leg_offset, "endpoint ID extent");
        if (run.tile_stride <= max_within_tile) {
            fail("endpoint tile_stride overlaps tile IDs");
        }
        checked_add(run.first_json_id,
                    checked_add(checked_mul(run.tile_count - 1, run.tile_stride,
                                            "endpoint tile ID extent"),
                                max_within_tile, "endpoint run ID extent"),
                    "endpoint run ID extent");
        for (const PipelineEndpointLegRun& leg : run.legs) {
            if (!leg.tag_emission_ordinal) continue;
            auto tag_it = tag_by_emission.find(*leg.tag_emission_ordinal);
            if (tag_it == tag_by_emission.end()
                    || tag_it->second->group_ordinal != run.group_ordinal) {
                fail("endpoint leg references missing tag run");
            }
            const TagRun& tag_run = *tag_it->second;
            const u64 tag_tile_end = checked_add(tag_run.first_tile,
                                                 tag_run.tile_count - 1,
                                                 "tag run tile range");
            if (!(tag_run.first_tile <= run.first_tile
                    && tile_end <= tag_tile_end)) {
                fail("endpoint tag run does not cover tile range");
            }
            for (u64 shard : run.shard_ordinals) {
                if (shard < tag_run.first_shard
                        || shard >= tag_run.first_shard + tag_run.shard_count) {
                    fail("endpoint tag run does not cover shard range");
                }
            }
            const u64 max_tile_local = run.tile_count - 1;
            const u64 max_shard_tag_local = run.shard_ordinals.size() - 1;
            const u64 max_tag_offset = checked_add(
                *leg.tag_offset,
                checked_add(
                    checked_mul(max_tile_local, *leg.tag_tile_stride,
                                "endpoint tile tag extent"),
                    checked_mul(max_shard_tag_local, *leg.tag_shard_stride,
                                "endpoint shard tag extent"),
                    "endpoint tag extent"),
                "endpoint tag extent");
            if (max_tag_offset >= tag_run.count) {
                fail("endpoint tag mapping exceeds allocation run");
            }
            checked_add(tag_run.first_tag, max_tag_offset, "endpoint tag value");
        }
    }
}

// ------------------------------------------------- affine dependency runs --

// _parse_affine_dependency_run (endpoint flag selects the endpoint field).
DependencyRun parse_affine_dependency_run(const Json& value,
                                          const std::string& path,
                                          bool with_endpoint) {
    std::vector<const char*> required = {"rank", "group_ordinal", "first_tile",
                                         "tile_count", "shard_ordinals"};
    if (with_endpoint) required.push_back("endpoint");
    required.insert(required.end(),
                    {"dependency_kind", "first_dependency_json_id",
                     "dependency_tile_stride", "dependency_shard_stride"});
    const Json& raw = mapping(value, path, required);
    DependencyRun out;
    const Json& shards_raw = p_array(raw["shard_ordinals"],
                                     path + ".shard_ordinals", true);
    out.shard_ordinals.reserve(shards_raw.size());
    for (std::size_t i = 0; i < shards_raw.size(); ++i) {
        out.shard_ordinals.push_back(p_uint(
            shards_raw[i],
            path + ".shard_ordinals[" + std::to_string(i) + "]"));
    }
    for (std::size_t i = 1; i < out.shard_ordinals.size(); ++i) {
        if (out.shard_ordinals[i - 1] >= out.shard_ordinals[i]) {
            fail(path + ".shard_ordinals must be strictly increasing");
        }
    }
    out.has_endpoint = with_endpoint;
    if (with_endpoint) {
        out.endpoint = p_enum(
            raw["endpoint"], path + ".endpoint",
            {"data_send", "data_recv", "fill_ack_send", "fill_ack_recv",
             "release_ack_send", "release_ack_recv", "body_marker"});
    }
    out.rank = p_rank(raw["rank"], path + ".rank");
    out.group_ordinal = p_uint(raw["group_ordinal"], path + ".group_ordinal");
    out.first_tile = p_positive(raw["first_tile"], path + ".first_tile");
    out.tile_count = p_positive(raw["tile_count"], path + ".tile_count");
    out.dependency_kind = p_enum(raw["dependency_kind"],
                                 path + ".dependency_kind",
                                 {"ordinary_node_terminal",
                                  "virtual_endpoint_terminal"});
    out.first_dependency_json_id =
        p_uint(raw["first_dependency_json_id"],
               path + ".first_dependency_json_id");
    out.dependency_tile_stride =
        p_uint(raw["dependency_tile_stride"], path + ".dependency_tile_stride");
    out.dependency_shard_stride =
        p_uint(raw["dependency_shard_stride"], path + ".dependency_shard_stride");
    return out;
}

// _validate_dependency_run_domain (rank argument unused, as in Python).
const PipelineGroup& validate_dependency_run_domain(
        u64 group_ordinal, u64 first_tile, u64 tile_count,
        const std::vector<u64>& shard_ordinals,
        const std::map<u64, const PipelineGroup*>& groups,
        const std::string& path) {
    auto group_it = groups.find(group_ordinal);
    if (group_it == groups.end()) {
        fail(path + " references an unknown group");
    }
    const PipelineGroup& group = *group_it->second;
    if (static_cast<i128>(first_tile) + tile_count - 1
            > static_cast<i128>(group.tile_axis.tile_count)) {
        fail(path + " exceeds its group tile axis");
    }
    for (u64 shard : shard_ordinals) {
        if (shard >= group.shards.size()) {
            fail(path + " references an unknown shard");
        }
    }
    return group;
}

// _validate_affine_dependency_extent.
void validate_affine_dependency_extent(u64 first_id, u64 tile_stride,
                                       u64 shard_stride, u64 tile_count,
                                       u64 shard_count,
                                       const std::string& path) {
    checked_add(first_id,
                checked_add(
                    checked_mul(tile_count - 1, tile_stride,
                                path + " tile dependency extent"),
                    checked_mul(shard_count - 1, shard_stride,
                                path + " shard dependency extent"),
                    path + " dependency extent"),
                path + " dependency extent");
}

// _validate_endpoint_dependency_runs.
void validate_endpoint_dependency_runs(
        const std::vector<DependencyRun>& dependency_runs,
        const std::vector<PipelineEndpointRun>& endpoint_runs,
        const std::vector<PipelineGroup>& groups) {
    const std::map<u64, const PipelineGroup*> group_by_id =
        groups_by_ordinal(groups);
    std::map<std::tuple<u64, u64, std::string, u64>,
             std::vector<std::pair<u64, u64>>>
        runs_by_position;
    for (const PipelineEndpointRun& run : endpoint_runs) {
        for (const PipelineEndpointLegRun& leg : run.legs) {
            for (u64 shard : run.shard_ordinals) {
                runs_by_position[{run.rank, run.group_ordinal, leg.endpoint,
                                  shard}]
                    .emplace_back(run.first_tile,
                                  run.first_tile + run.tile_count - 1);
            }
        }
    }
    using DupKey = std::tuple<u64, u64, u64, u64, std::vector<u64>,
                              std::string, std::string, u64, u64, u64>;
    std::set<DupKey> seen;
    for (std::size_t index = 0; index < dependency_runs.size(); ++index) {
        const DependencyRun& run = dependency_runs[index];
        const std::string path =
            "endpoint_dependency_runs[" + std::to_string(index) + "]";
        const PipelineGroup& group = validate_dependency_run_domain(
            run.group_ordinal, run.first_tile, run.tile_count,
            run.shard_ordinals, group_by_id, path);
        for (u64 shard_index : run.shard_ordinals) {
            const PipelineShardRule& shard =
                group.shards[static_cast<std::size_t>(shard_index)];
            const PipelineTileAxis& axis = group.tile_axis;
            const u64 active_count =
                axis.unit == "source_tokens"
                    ? axis.tile_count
                    : std::min(axis.tile_count,
                               ceil_div_u64(shard.total_data_bytes,
                                            axis.units_per_tile));
            const i128 run_tile_end =
                static_cast<i128>(run.first_tile) + run.tile_count - 1;
            if (run.endpoint != "body_marker"
                    && run_tile_end > static_cast<i128>(active_count)) {
                fail(path + " exceeds this shard's active tile cap");
            }
            if (endpoint_owner_rank(shard, run.endpoint) != run.rank) {
                fail(path + ".rank disagrees with endpoint route");
            }
            auto pieces = runs_by_position.find(
                {run.rank, run.group_ordinal, run.endpoint, shard_index});
            bool covered = false;
            if (pieces != runs_by_position.end()) {
                for (const auto& interval : pieces->second) {
                    if (interval.first <= run.first_tile
                            && run_tile_end
                                <= static_cast<i128>(interval.second)) {
                        covered = true;
                        break;
                    }
                }
            }
            if (!covered) {
                fail(path + " has no matching emitted endpoint run");
            }
        }
        validate_affine_dependency_extent(
            run.first_dependency_json_id, run.dependency_tile_stride,
            run.dependency_shard_stride, run.tile_count,
            static_cast<u64>(run.shard_ordinals.size()), path);
        const DupKey key{run.rank, run.group_ordinal, run.first_tile,
                         run.tile_count, run.shard_ordinals, run.endpoint,
                         run.dependency_kind, run.first_dependency_json_id,
                         run.dependency_tile_stride,
                         run.dependency_shard_stride};
        if (!seen.insert(key).second) {
            fail(path + " duplicates an earlier run");
        }
    }
}

// _validate_consume_completion_runs.
void validate_consume_completion_runs(
        const std::vector<PipelineConsumeCompletionRun>& completion_runs,
        const std::vector<PipelineGroup>& groups, u64 identity_rank) {
    const std::map<u64, const PipelineGroup*> group_by_id =
        groups_by_ordinal(groups);
    std::map<std::pair<u64, u64>, std::vector<std::pair<u64, u64>>> covered;
    for (std::size_t index = 0; index < completion_runs.size(); ++index) {
        const PipelineConsumeCompletionRun& run = completion_runs[index];
        const std::string path =
            "consume_completion_runs[" + std::to_string(index) + "]";
        const PipelineGroup& group = validate_dependency_run_domain(
            run.group_ordinal, run.first_tile, run.tile_count,
            run.shard_ordinals, group_by_id, path);
        if (run.rank != identity_rank) {
            fail(path + ".rank must equal the descriptor buffer owner rank");
        }
        for (u64 shard_index : run.shard_ordinals) {
            if (group.shards[static_cast<std::size_t>(shard_index)].target_rank
                    != run.rank) {
                fail(path + ".rank must be the target-side owner rank");
            }
            covered[{run.group_ordinal, shard_index}].emplace_back(
                run.first_tile, run.first_tile + run.tile_count - 1);
        }
        validate_affine_dependency_extent(
            run.first_dependency_json_id, run.dependency_tile_stride,
            run.dependency_shard_stride, run.tile_count,
            static_cast<u64>(run.shard_ordinals.size()), path);
    }
    for (const PipelineGroup& group : groups) {
        const PipelineTileAxis& axis = group.tile_axis;
        for (std::size_t shard_index = 0; shard_index < group.shards.size();
             ++shard_index) {
            const PipelineShardRule& shard = group.shards[shard_index];
            if (shard.target_rank != identity_rank) continue;
            u64 active_tiles;
            if (axis.unit == "source_tokens") {
                active_tiles = shard.bytes_per_unit ? axis.tile_count : 0;
            } else {
                active_tiles = std::min(
                    axis.tile_count,
                    ceil_div_u64(shard.total_data_bytes, axis.units_per_tile));
            }
            auto intervals_it = covered.find(
                {group.group_ordinal, static_cast<u64>(shard_index)});
            std::vector<std::pair<u64, u64>> intervals;
            if (intervals_it != covered.end()) intervals = intervals_it->second;
            std::sort(intervals.begin(), intervals.end());
            i128 next_tile = 1;
            for (const auto& interval : intervals) {
                if (static_cast<i128>(interval.first) != next_tile) {
                    fail("consume completion runs have a gap or overlap for group "
                         + std::to_string(group.group_ordinal) + ", shard "
                         + std::to_string(shard_index));
                }
                next_tile =
                    static_cast<i128>(interval.second) + 1;
            }
            if (next_tile != static_cast<i128>(active_tiles) + 1) {
                fail("consume completion runs do not cover every active target "
                     "tile for group " + std::to_string(group.group_ordinal)
                     + ", shard " + std::to_string(shard_index));
            }
        }
    }
}

// ------------------------------------------------------------ projection --

// _parse_projection_share.
std::optional<ProjectionShare> parse_projection_share(const Json& value,
                                                      const std::string& path) {
    if (value.is_null()) return std::nullopt;
    const Json& raw = mapping(value, path,
        {"rule", "total_units", "total_tiles", "share_index_first",
         "tile_bytes"});
    ProjectionShare out;
    out.rule = p_enum(raw["rule"], path + ".rule", {kShareRule});
    out.total_units = p_uint(raw["total_units"], path + ".total_units");
    out.total_tiles = p_positive(raw["total_tiles"], path + ".total_tiles");
    out.share_index_first = p_uint(raw["share_index_first"],
                                   path + ".share_index_first");
    out.tile_bytes = p_positive(raw["tile_bytes"], path + ".tile_bytes");
    return out;
}

// _completion_order.
std::pair<std::optional<i64>, i64> parse_completion_order(
        const Json& raw, const std::string& path, u64 tile_count) {
    std::optional<i64> first;
    if (!raw["completion_order_first"].is_null()) {
        first = p_int64(raw["completion_order_first"],
                        path + ".completion_order_first");
    }
    const i64 stride =
        p_int64(raw["completion_order_stride"], path + ".completion_order_stride");
    if (!first) {
        if (stride != 0) {
            fail(path + " null completion_order_first requires zero stride");
        }
    } else {
        const i128 last = static_cast<i128>(*first)
            + static_cast<i128>(stride)
                * (static_cast<i128>(tile_count) - 1);
        if (last < static_cast<i128>(kI64Min)
                || last > static_cast<i128>(kI64Max)) {
            fail(path + " completion-order progression overflows int64");
        }
    }
    return {first, stride};
}

// _parse_projection_fill.
ProjectionFill parse_projection_fill(const Json& value,
                                     const std::string& path, u64 tile_count,
                                     const PipelineProtocol& protocol) {
    (void)protocol;
    const Json& raw = mapping(value, path,
        {"leg_id_prefix", "first_index", "fill_order_first",
         "fill_order_stride", "completion_order_first", "completion_order_stride",
         "route", "bytes_per_tile", "latency_ns", "start_deps", "gate_id",
         "gate_capacity"});
    ProjectionFill out;
    std::tie(out.completion_order_first, out.completion_order_stride) =
        parse_completion_order(raw, path, tile_count);
    out.leg_id_prefix = p_string(raw["leg_id_prefix"], path + ".leg_id_prefix");
    out.first_index = p_uint(raw["first_index"], path + ".first_index");
    out.fill_order_first = p_uint(raw["fill_order_first"],
                                  path + ".fill_order_first");
    out.fill_order_stride = p_positive(raw["fill_order_stride"],
                                       path + ".fill_order_stride");
    out.route = string_array(raw["route"], path + ".route");
    out.bytes_per_tile = p_positive(raw["bytes_per_tile"],
                                    path + ".bytes_per_tile");
    out.latency_ns = p_uint(raw["latency_ns"], path + ".latency_ns");
    out.start_deps = string_array(raw["start_deps"], path + ".start_deps");
    out.gate_id = p_string(raw["gate_id"], path + ".gate_id");
    out.gate_capacity = p_positive(raw["gate_capacity"], path + ".gate_capacity",
                                   kU32Max);
    return out;
}

// _parse_projection_consume.
ProjectionConsume parse_projection_consume(const Json& value,
                                           const std::string& path,
                                           u64 tile_count) {
    const Json& raw = mapping(value, path,
        {"leg_id_prefix", "first_index", "consume_order_first",
         "consume_order_stride", "completion_order_first",
         "completion_order_stride", "kind", "route", "group",
         "latency_first_ns", "latency_rest_ns", "consume_deps",
         "consume_start_deps", "gate_id", "compute_share", "hbm_share"});
    ProjectionConsume out;
    std::tie(out.completion_order_first, out.completion_order_stride) =
        parse_completion_order(raw, path, tile_count);
    out.leg_id_prefix = p_string(raw["leg_id_prefix"], path + ".leg_id_prefix");
    out.first_index = p_uint(raw["first_index"], path + ".first_index");
    out.consume_order_first = p_uint(raw["consume_order_first"],
                                     path + ".consume_order_first");
    out.consume_order_stride = p_positive(raw["consume_order_stride"],
                                          path + ".consume_order_stride");
    out.kind = p_enum(raw["kind"], path + ".kind", {"fixed", "compute"});
    if (!raw["group"].is_null()) {
        out.group = p_string(raw["group"], path + ".group");
    }
    out.compute_share = parse_projection_share(raw["compute_share"],
                                               path + ".compute_share");
    out.hbm_share = parse_projection_share(raw["hbm_share"], path + ".hbm_share");
    out.route = string_array(raw["route"], path + ".route");
    out.latency_first_ns = p_uint(raw["latency_first_ns"],
                                  path + ".latency_first_ns");
    out.latency_rest_ns = p_uint(raw["latency_rest_ns"],
                                 path + ".latency_rest_ns");
    if (out.kind == "fixed") {
        if (!out.route.empty() || out.group
                || out.latency_first_ns || out.latency_rest_ns
                || out.compute_share || out.hbm_share) {
            fail(path + " fixed consume must have empty route, null group/shares,"
                 " and zero latency");
        }
    } else {
        if (out.route.empty() || !out.group || *out.group != "decode") {
            fail(path + " compute consume requires a route and decode group");
        }
        if (!out.compute_share || !out.hbm_share) {
            fail(path
                 + " compute consume requires separate compute_share and hbm_share");
        }
        const char* names[2] = {"compute_share", "hbm_share"};
        const std::optional<ProjectionShare>* shares[2] = {&out.compute_share,
                                                           &out.hbm_share};
        for (int i = 0; i < 2; ++i) {
            const ProjectionShare& share = **shares[i];
            if (share.total_tiles != out.compute_share->total_tiles
                    || static_cast<i128>(share.share_index_first) + tile_count
                        > static_cast<i128>(share.total_tiles)) {
                fail(path + "." + names[i]
                     + " tile range is inconsistent with the chain");
            }
        }
        if (out.compute_share->tile_bytes != out.hbm_share->tile_bytes
                || out.compute_share->share_index_first
                    != out.hbm_share->share_index_first) {
            fail(path
                 + " compute and HBM shares must use the same global byte positions");
        }
    }
    out.consume_deps = string_array(raw["consume_deps"], path + ".consume_deps");
    out.consume_start_deps = string_array(raw["consume_start_deps"],
                                          path + ".consume_start_deps");
    out.gate_id = p_string(raw["gate_id"], path + ".gate_id");
    return out;
}

// _progression_ranges_disjoint.
void progression_ranges_disjoint(const std::vector<Progression>& entries,
                                 const std::string& label) {
    for (std::size_t i = 0; i < entries.size(); ++i) {
        for (std::size_t j = i + 1; j < entries.size(); ++j) {
            if (progressions_intersect(entries[i], entries[j])) {
                fail("projection " + label + " ranges overlap");
            }
        }
    }
}

// _validate_projection.
void validate_projection(const std::vector<ProjectionChain>& chains,
                         const std::vector<PipelineGroup>& groups,
                         const PipelineProtocol& protocol, u64 owner_rank) {
    const std::map<u64, const PipelineGroup*> group_by_id =
        groups_by_ordinal(groups);
    std::set<std::pair<u64, u64>> seen_group_rank;
    std::map<std::string, u64> gate_capacity;
    std::vector<Progression> fill_orders;
    std::vector<Progression> consume_orders;
    std::map<std::string,
             std::vector<std::tuple<u64, u64, std::string>>>
        leg_ranges;
    std::map<u64, std::vector<std::tuple<u64, u64, u64>>> per_rank;
    for (std::size_t index = 0; index < chains.size(); ++index) {
        const ProjectionChain& chain = chains[index];
        const std::string path =
            "projection.chains[" + std::to_string(index) + "]";
        auto group_it = group_by_id.find(chain.group_ordinal);
        if (group_it == group_by_id.end()) {
            fail(path + " references unknown group");
        }
        const PipelineGroup& group = *group_it->second;
        if (chain.tile_count != group.tile_axis.tile_count) {
            fail(path + " tile_count must match its group");
        }
        const std::pair<u64, u64> key{chain.group_ordinal, chain.rank};
        if (!seen_group_rank.insert(key).second) {
            fail(path + " duplicates a group/rank chain");
        }
        std::set<u64> routed_targets;
        for (const PipelineShardRule& shard : group.shards) {
            if (shard.total_data_bytes > 0) {
                routed_targets.insert(shard.target_rank);
            }
        }
        if (!routed_targets.count(chain.rank)) {
            fail(path + " rank has no charged target shard");
        }
        if (chain.rank != owner_rank) {
            fail(path + " rank must equal descriptor buffer-domain owner rank");
        }
        const ProjectionFill& fill = chain.fill;
        const ProjectionConsume& consume = chain.consume;
        if (fill.first_index != consume.first_index) {
            fail(path + " fill and consume index ranges must align");
        }
        if (fill.route.empty()) {
            fail(path + ".fill route must not be empty");
        }
        if (fill.gate_capacity != protocol.buffer_slots) {
            fail(path
                 + ".fill gate capacity disagrees with the double-buffer protocol");
        }
        if (fill.gate_id != consume.gate_id) {
            fail(path + " fill/consume gate IDs must match");
        }
        auto prior_capacity = gate_capacity.find(fill.gate_id);
        if (prior_capacity == gate_capacity.end()) {
            gate_capacity[fill.gate_id] = fill.gate_capacity;
        } else if (prior_capacity->second != fill.gate_capacity) {
            fail("a projection gate ID has conflicting capacities");
        }
        checked_add(fill.first_index, chain.tile_count,
                    "projection fill ID index range");
        checked_add(consume.first_index, chain.tile_count,
                    "projection consume ID index range");
        leg_ranges[fill.leg_id_prefix].emplace_back(
            fill.first_index, fill.first_index + chain.tile_count,
            path + ".fill");
        leg_ranges[consume.leg_id_prefix].emplace_back(
            consume.first_index, consume.first_index + chain.tile_count,
            path + ".consume");
        fill_orders.push_back(
            Progression{0, fill.fill_order_first, fill.fill_order_stride,
                        chain.tile_count});
        consume_orders.push_back(
            Progression{0, consume.consume_order_first,
                        consume.consume_order_stride, chain.tile_count});
        per_rank[chain.rank].emplace_back(chain.group_ordinal, fill.first_index,
                                          chain.tile_count);
    }
    for (const auto& entry : leg_ranges) {
        std::vector<std::tuple<u64, u64, std::string>> ranges = entry.second;
        std::sort(ranges.begin(), ranges.end());
        for (std::size_t i = 1; i < ranges.size(); ++i) {
            if (std::get<0>(ranges[i]) < std::get<1>(ranges[i - 1])) {
                fail("projection leg ID ranges overlap for prefix '"
                     + entry.first + "'");
            }
        }
    }
    fill_orders.insert(fill_orders.end(), consume_orders.begin(),
                       consume_orders.end());
    progression_ranges_disjoint(fill_orders, "registration-order");
    for (auto& rank_entry : per_rank) {
        std::vector<std::tuple<u64, u64, u64>>& segments = rank_entry.second;
        std::sort(segments.begin(), segments.end());
        if (std::get<1>(segments[0]) != 0) {
            fail("projection rank " + std::to_string(rank_entry.first)
                 + " first chain must start at index zero");
        }
        bool have_previous = false;
        u128 previous_end = 0;
        for (const auto& segment : segments) {
            if (have_previous
                    && static_cast<i128>(std::get<1>(segment))
                        != static_cast<i128>(previous_end)) {
                fail("projection rank " + std::to_string(rank_entry.first)
                     + " group chains are not contiguous");
            }
            previous_end = static_cast<u128>(std::get<1>(segment))
                + std::get<2>(segment);
            have_previous = true;
        }
    }
}

// -------------------------------------------------------------- anchors --

// _parse_anchor.
Anchor parse_anchor(const Json& value, const std::string& path) {
    const Json& raw = mapping(value, path,
                              {"role", "rank", "binding", "node_id", "selector"});
    Anchor anchor;
    anchor.role = p_enum(raw["role"], path + ".role",
                         {"start", "frontier", "service", "settle"});
    anchor.rank = p_rank(raw["rank"], path + ".rank");
    anchor.binding =
        p_enum(raw["binding"], path + ".binding",
               {"physical_endpoint_slot", "descriptor_cursor_selector"});
    anchor.node_id = read_nullable_uint(raw["node_id"], path + ".node_id");
    const Json& selector_raw =
        mapping(raw["selector"], path + ".selector", {"kind"},
                {"group_ordinal", "tile_ordinal", "shard_ordinal", "endpoint",
                 "existing_gate", "tag_allocation_emission_ordinal",
                 "tag_offset"});
    AnchorSelector& selector = anchor.selector;
    selector.kind = p_enum(selector_raw["kind"], path + ".selector.kind",
        {"stage_start", "tile_endpoint_leg", "tile_endpoint_terminal",
         "internal_marker_done", "existing_service_gate",
         "all_obligations_settled"});
    static const char* kUintKeys[4] = {"group_ordinal", "tile_ordinal",
                                       "shard_ordinal",
                                       "tag_allocation_emission_ordinal"};
    for (const char* key : kUintKeys) {
        auto member = selector_raw.find(key);
        if (member == selector_raw.end()) continue;
        const std::string member_path = path + ".selector." + key;
        const u64 parsed = p_uint(*member, member_path);
        if (key == std::string("group_ordinal")) selector.group_ordinal = parsed;
        else if (key == std::string("tile_ordinal")) selector.tile_ordinal = parsed;
        else if (key == std::string("shard_ordinal")) selector.shard_ordinal = parsed;
        else selector.tag_allocation_emission_ordinal = parsed;
    }
    auto member = selector_raw.find("tag_offset");
    if (member != selector_raw.end()) {
        selector.tag_offset = p_uint(*member, path + ".selector.tag_offset",
                                     kU32Max);
    }
    member = selector_raw.find("endpoint");
    if (member != selector_raw.end()) {
        selector.endpoint = p_enum(
            *member, path + ".selector.endpoint",
            {"data_send", "data_recv", "fill_ack_send", "fill_ack_recv",
             "release_ack_send", "release_ack_recv", "body_marker"});
    }
    member = selector_raw.find("existing_gate");
    if (member != selector_raw.end()) {
        selector.existing_gate =
            p_string(*member, path + ".selector.existing_gate");
    }
    return anchor;
}

// _validate_anchors.
void validate_anchors(const std::vector<Anchor>& anchors,
                      const std::vector<PipelineGroup>& groups,
                      const std::vector<TagRun>& runs,
                      const std::vector<PipelineEndpointRun>& endpoint_runs) {
    const std::map<u64, const PipelineGroup*> group_by_id =
        groups_by_ordinal(groups);
    std::map<u64, const TagRun*> runs_by_emission;
    for (const TagRun& run : runs) runs_by_emission[run.emission_ordinal] = &run;
    std::map<std::string, std::size_t> role_counts;
    std::set<std::pair<u64, u64>> physical_slots;
    std::vector<std::pair<u64, AnchorSelector>> selectors;
    for (const Anchor& anchor : anchors) {
        role_counts[anchor.role] += 1;
        const AnchorSelector& sel = anchor.selector;
        const std::pair<u64, AnchorSelector> selector_key{anchor.rank, sel};
        for (const auto& existing : selectors) {
            if (existing == selector_key) {
                fail("anchor selectors must be unique per rank");
            }
        }
        selectors.push_back(selector_key);
        const bool endpoint_selector = sel.kind == "tile_endpoint_leg"
            || sel.kind == "tile_endpoint_terminal"
            || sel.kind == "internal_marker_done";
        if (anchor.binding == "physical_endpoint_slot") {
            if (anchor.role != "start" && anchor.role != "frontier") {
                fail("only start/frontier anchors may bind physical endpoint slots");
            }
            if (!anchor.node_id) {
                fail("physical endpoint anchor requires node_id");
            }
            const std::pair<u64, u64> slot_key{anchor.rank, *anchor.node_id};
            if (!physical_slots.insert(slot_key).second) {
                fail("physical endpoint slots must be unique");
            }
            if (endpoint_selector) {
                const bool marker_selector =
                    sel.kind == "internal_marker_done";
                bool complete = sel.group_ordinal && sel.tile_ordinal
                    && sel.shard_ordinal && sel.endpoint;
                if (!marker_selector) {
                    complete = complete && sel.tag_allocation_emission_ordinal
                        && sel.tag_offset;
                }
                if (!complete || *sel.tile_ordinal == 0) {
                    fail("physical endpoint anchor requires complete one-based selector");
                }
                if (marker_selector
                        && (*sel.endpoint != "body_marker"
                            || anchor.role != "frontier"
                            || sel.tag_allocation_emission_ordinal
                            || sel.tag_offset)) {
                    fail("internal_marker_done selector must be a frontier"
                         " body_marker without tag fields");
                }
                if (!marker_selector && *sel.endpoint == "body_marker") {
                    fail("body_marker anchors must use internal_marker_done");
                }
                auto group_it = group_by_id.find(*sel.group_ordinal);
                if (group_it == group_by_id.end()
                        || *sel.shard_ordinal
                            >= group_it->second->shards.size()
                        || *sel.tile_ordinal
                            > group_it->second->tile_axis.tile_count) {
                    fail("physical endpoint selector is outside its group");
                }
                std::optional<u64> tag_offset;
                if (marker_selector) {
                    tag_offset = std::nullopt;
                } else {
                    auto run_it = runs_by_emission.find(
                        *sel.tag_allocation_emission_ordinal);
                    if (run_it == runs_by_emission.end()
                            || run_it->second->group_ordinal
                                != *sel.group_ordinal) {
                        fail("physical endpoint selector references missing"
                             " tag run");
                    }
                    const TagRun& run = *run_it->second;
                    if (*sel.tag_offset >= run.count
                            || !(run.first_tile <= *sel.tile_ordinal
                                 && *sel.tile_ordinal
                                     < run.first_tile + run.tile_count)
                            || !(run.first_shard <= *sel.shard_ordinal
                                 && *sel.shard_ordinal
                                     < run.first_shard + run.shard_count)) {
                        fail("physical endpoint selector is outside tag run");
                    }
                    tag_offset = *sel.tag_offset;
                }
                struct Match {
                    std::size_t run_index;
                    std::size_t leg_index;
                };
                std::vector<Match> matches;
                for (std::size_t run_ordinal = 0;
                     run_ordinal < endpoint_runs.size(); ++run_ordinal) {
                    const PipelineEndpointRun& endpoint_run =
                        endpoint_runs[run_ordinal];
                    if (endpoint_run.group_ordinal != *sel.group_ordinal
                            || !(endpoint_run.first_tile <= *sel.tile_ordinal
                                 && *sel.tile_ordinal
                                     < static_cast<i128>(
                                           endpoint_run.first_tile)
                                         + endpoint_run.tile_count)
                            || std::find(endpoint_run.shard_ordinals.begin(),
                                         endpoint_run.shard_ordinals.end(),
                                         *sel.shard_ordinal)
                                == endpoint_run.shard_ordinals.end()) {
                        continue;
                    }
                    for (std::size_t leg_ordinal = 0;
                         leg_ordinal < endpoint_run.legs.size(); ++leg_ordinal) {
                        const PipelineEndpointLegRun& leg =
                            endpoint_run.legs[leg_ordinal];
                        if (leg.endpoint == *sel.endpoint
                                && (marker_selector
                                    || (leg.tag_emission_ordinal
                                        && *leg.tag_emission_ordinal
                                            == *sel.tag_allocation_emission_ordinal))) {
                            matches.push_back(Match{run_ordinal, leg_ordinal});
                        }
                    }
                }
                if (matches.size() != 1) {
                    fail("physical endpoint selector must identify one endpoint"
                         " run leg");
                }
                const PipelineEndpointRun& endpoint_run =
                    endpoint_runs[matches[0].run_index];
                const PipelineEndpointLegRun& leg =
                    endpoint_run.legs[matches[0].leg_index];
                if (endpoint_run.rank != anchor.rank) {
                    fail("physical endpoint anchor rank disagrees with endpoint"
                         " run");
                }
                const u64 local_tile = *sel.tile_ordinal - endpoint_run.first_tile;
                const u64 local_shard = static_cast<u64>(
                    std::find(endpoint_run.shard_ordinals.begin(),
                              endpoint_run.shard_ordinals.end(),
                              *sel.shard_ordinal)
                    - endpoint_run.shard_ordinals.begin());
                u64 expected_node_id = checked_add(
                    endpoint_run.first_json_id,
                    checked_add(
                        checked_mul(local_tile, endpoint_run.tile_stride,
                                    "anchor tile ID offset"),
                        checked_mul(local_shard, endpoint_run.shard_stride,
                                    "anchor shard ID offset"),
                        "anchor ID offset"),
                    "anchor ID");
                expected_node_id = checked_add(expected_node_id,
                                               leg.json_offset, "anchor leg ID");
                if (expected_node_id != *anchor.node_id) {
                    fail("physical endpoint node_id disagrees with endpoint ID"
                         " mapping");
                }
                if (marker_selector) {
                    if (leg.tag_emission_ordinal) {
                        fail("internal marker endpoint must not allocate a tag");
                    }
                } else {
                    const u64 expected_tag_offset = checked_add(
                        *leg.tag_offset,
                        checked_add(
                            checked_mul(local_tile, *leg.tag_tile_stride,
                                        "anchor tile tag offset"),
                            checked_mul(local_shard, *leg.tag_shard_stride,
                                        "anchor shard tag offset"),
                            "anchor tag offset"),
                        "anchor tag offset");
                    if (expected_tag_offset != *sel.tag_offset) {
                        fail("physical endpoint tag selector disagrees with"
                             " endpoint mapping");
                    }
                }
            } else {
                fail("physical anchor must bind a concrete endpoint selector");
            }
        } else {
            if (anchor.node_id) {
                fail("cursor selector must not carry node_id");
            }
            if (anchor.role == "service" && sel.kind != "existing_service_gate") {
                fail("cursor service anchor must select existing_service_gate");
            }
            if (anchor.role == "settle"
                    && sel.kind != "all_obligations_settled") {
                fail("cursor settle anchor must select all_obligations_settled");
            }
            if ((anchor.role == "service" || anchor.role == "settle")
                    && anchor.binding != "descriptor_cursor_selector") {
                fail("service/settle anchors must be cursor-only");
            }
        }
    }
    for (const char* singleton_role : {"service", "settle"}) {
        auto count = role_counts.find(singleton_role);
        if (count == role_counts.end() || count->second != 1) {
            fail(std::string("descriptor needs exactly one ") + singleton_role
                 + " anchor");
        }
    }
}

// ----------------------------------------------- endpoint ID target checks --

// _validate_endpoint_id_targets.
void validate_endpoint_id_targets(
        const std::vector<PipelineEndpointRun>& endpoint_runs,
        const std::vector<IdRange>& id_ranges,
        const std::vector<Anchor>& anchors) {
    const EndpointIdIndex index(id_ranges, anchors);
    for (const Anchor& anchor : anchors) {
        if (anchor.binding != "physical_endpoint_slot" || !anchor.node_id) {
            continue;
        }
        if (index.anchor_hits_hole(anchor.rank, *anchor.node_id)) {
            fail("physical endpoint anchor aliases a virtual ID hole");
        }
    }
    struct SweepEntry {
        u64 first;
        u128 last;
        Progression progression;
    };
    std::vector<SweepEntry> sweep;
    for (std::size_t run_index = 0; run_index < endpoint_runs.size();
         ++run_index) {
        const PipelineEndpointRun& run = endpoint_runs[run_index];
        for (std::size_t shard_index = 0;
             shard_index < run.shard_ordinals.size(); ++shard_index) {
            for (std::size_t leg_index = 0; leg_index < run.legs.size();
                 ++leg_index) {
                const PipelineEndpointLegRun& leg = run.legs[leg_index];
                const u64 first = checked_add(
                    run.first_json_id,
                    checked_add(
                        checked_mul(static_cast<u64>(shard_index),
                                    run.shard_stride,
                                    "endpoint shard progression"),
                        leg.json_offset, "endpoint leg progression"),
                    "endpoint progression");
                const Progression progression{run.rank, first, run.tile_stride,
                                              run.tile_count};
                if (index.covered_count(run.rank, first, run.tile_stride,
                                        run.tile_count)
                        != static_cast<u128>(run.tile_count)) {
                    fail("endpoint_runs[" + std::to_string(run_index)
                         + "] shard " + std::to_string(shard_index) + " leg "
                         + std::to_string(leg_index)
                         + " maps an ID neither to a virtual hole nor a"
                           " physical anchor");
                }
                sweep.push_back(SweepEntry{
                    first,
                    static_cast<u128>(first)
                        + static_cast<u128>(run.tile_stride)
                            * (run.tile_count - 1),
                    progression});
            }
        }
    }
    std::stable_sort(sweep.begin(), sweep.end(),
                     [](const SweepEntry& a, const SweepEntry& b) {
                         return a.first < b.first;
                     });
    std::vector<std::pair<u128, Progression>> actives;
    for (const SweepEntry& entry : sweep) {
        std::vector<std::pair<u128, Progression>> kept;
        kept.reserve(actives.size() + 1);
        for (const auto& active : actives) {
            if (active.first < entry.first) continue;  // retired
            if (progressions_intersect(active.second, entry.progression)) {
                fail("affine endpoint IDs overlap");
            }
            kept.push_back(active);
        }
        kept.emplace_back(entry.last, entry.progression);
        actives = std::move(kept);
    }
}

// ---------------------------------------------------------- wire builders --

// _group_wire.
Json group_wire(const PipelineGroup& group) {
    Json out;
    out["group_ordinal"] = group.group_ordinal;
    out["pass_or_window_ordinal"] = group.pass_or_window_ordinal;
    out["source_ordinal"] = group.source_ordinal;
    out["source_key"] = group.source_key;
    out["layer_start"] = group.layer_start;
    out["layer_end"] = group.layer_end;
    Json axis;
    axis["unit"] = group.tile_axis.unit;
    axis["total_units"] = group.tile_axis.total_units;
    axis["units_per_tile"] = group.tile_axis.units_per_tile;
    axis["tile_count"] = group.tile_axis.tile_count;
    axis["tail_units"] = group.tile_axis.tail_units;
    axis["global_tile_first"] = group.tile_axis.global_tile_first;
    out["tile_axis"] = std::move(axis);
    Json shards = Json::array();
    for (const PipelineShardRule& shard : group.shards) {
        Json shard_json;
        shard_json["shard_ordinal"] = shard.shard_ordinal;
        shard_json["source_rank"] = shard.source_rank;
        shard_json["target_rank"] = shard.target_rank;
        shard_json["total_data_bytes"] = shard.total_data_bytes;
        shard_json["bytes_per_unit"] = shard.bytes_per_unit;
        shard_json["noc_path"] = shard.noc_path;
        shards.push_back(std::move(shard_json));
    }
    out["shards"] = std::move(shards);
    return out;
}

// _endpoint_run_wire.
Json endpoint_run_wire(const PipelineEndpointRun& run) {
    Json out;
    out["rank"] = run.rank;
    out["group_ordinal"] = run.group_ordinal;
    out["first_tile"] = run.first_tile;
    out["tile_count"] = run.tile_count;
    out["shard_ordinals"] = run.shard_ordinals;
    out["first_json_id"] = run.first_json_id;
    out["tile_stride"] = run.tile_stride;
    out["shard_stride"] = run.shard_stride;
    out["initial_parent_json_id"] =
        run.initial_parent_json_id ? Json(*run.initial_parent_json_id)
                                   : Json(nullptr);
    Json context;
    context["request_id"] = run.legacy_context.request_id;
    context["stage"] = run.legacy_context.stage;
    context["generation"] = run.legacy_context.generation;
    out["legacy_context"] = std::move(context);
    Json legs = Json::array();
    for (const PipelineEndpointLegRun& leg : run.legs) {
        Json leg_json;
        leg_json["endpoint"] = leg.endpoint;
        leg_json["json_offset"] = leg.json_offset;
        leg_json["tag_emission_ordinal"] =
            leg.tag_emission_ordinal ? Json(*leg.tag_emission_ordinal)
                                     : Json(nullptr);
        leg_json["tag_offset"] =
            leg.tag_offset ? Json(*leg.tag_offset) : Json(nullptr);
        leg_json["tag_tile_stride"] =
            leg.tag_tile_stride ? Json(*leg.tag_tile_stride) : Json(nullptr);
        leg_json["tag_shard_stride"] =
            leg.tag_shard_stride ? Json(*leg.tag_shard_stride) : Json(nullptr);
        legs.push_back(std::move(leg_json));
    }
    out["legs"] = std::move(legs);
    return out;
}

// _endpoint_dependency_run_wire.
Json dependency_run_wire(const DependencyRun& run) {
    Json out;
    out["rank"] = run.rank;
    out["group_ordinal"] = run.group_ordinal;
    out["first_tile"] = run.first_tile;
    out["tile_count"] = run.tile_count;
    out["shard_ordinals"] = run.shard_ordinals;
    if (run.has_endpoint) out["endpoint"] = run.endpoint;
    out["dependency_kind"] = run.dependency_kind;
    out["first_dependency_json_id"] = run.first_dependency_json_id;
    out["dependency_tile_stride"] = run.dependency_tile_stride;
    out["dependency_shard_stride"] = run.dependency_shard_stride;
    return out;
}

// _consume_completion_run_wire.
Json completion_run_wire(const PipelineConsumeCompletionRun& run) {
    Json out;
    out["rank"] = run.rank;
    out["group_ordinal"] = run.group_ordinal;
    out["first_tile"] = run.first_tile;
    out["tile_count"] = run.tile_count;
    out["shard_ordinals"] = run.shard_ordinals;
    out["dependency_kind"] = run.dependency_kind;
    out["first_dependency_json_id"] = run.first_dependency_json_id;
    out["dependency_tile_stride"] = run.dependency_tile_stride;
    out["dependency_shard_stride"] = run.dependency_shard_stride;
    return out;
}

// _projection_share_wire.
Json projection_share_wire(const ProjectionShare& share) {
    Json out;
    out["rule"] = share.rule;
    out["total_units"] = share.total_units;
    out["total_tiles"] = share.total_tiles;
    out["share_index_first"] = share.share_index_first;
    out["tile_bytes"] = share.tile_bytes;
    return out;
}

// _projection_wire (from the parsed chains).
Json projection_to_wire(u64 version,
                        const std::vector<ProjectionChain>& chains) {
    Json chains_json = Json::array();
    for (const ProjectionChain& chain : chains) {
        const ProjectionFill& fill = chain.fill;
        const ProjectionConsume& consume = chain.consume;
        Json chain_json;
        chain_json["group_ordinal"] = chain.group_ordinal;
        chain_json["rank"] = chain.rank;
        chain_json["tile_count"] = chain.tile_count;
        Json fill_json;
        fill_json["leg_id_prefix"] = fill.leg_id_prefix;
        fill_json["first_index"] = fill.first_index;
        fill_json["fill_order_first"] = fill.fill_order_first;
        fill_json["fill_order_stride"] = fill.fill_order_stride;
        fill_json["completion_order_first"] =
            fill.completion_order_first ? Json(*fill.completion_order_first)
                                        : Json(nullptr);
        fill_json["completion_order_stride"] = fill.completion_order_stride;
        fill_json["route"] = fill.route;
        fill_json["bytes_per_tile"] = fill.bytes_per_tile;
        fill_json["latency_ns"] = fill.latency_ns;
        fill_json["start_deps"] = fill.start_deps;
        fill_json["gate_id"] = fill.gate_id;
        fill_json["gate_capacity"] = fill.gate_capacity;
        chain_json["fill"] = std::move(fill_json);
        Json consume_json;
        consume_json["leg_id_prefix"] = consume.leg_id_prefix;
        consume_json["first_index"] = consume.first_index;
        consume_json["consume_order_first"] = consume.consume_order_first;
        consume_json["consume_order_stride"] = consume.consume_order_stride;
        consume_json["completion_order_first"] =
            consume.completion_order_first
                ? Json(*consume.completion_order_first)
                : Json(nullptr);
        consume_json["completion_order_stride"] =
            consume.completion_order_stride;
        consume_json["kind"] = consume.kind;
        consume_json["route"] = consume.route;
        consume_json["group"] =
            consume.group ? Json(*consume.group) : Json(nullptr);
        consume_json["latency_first_ns"] = consume.latency_first_ns;
        consume_json["latency_rest_ns"] = consume.latency_rest_ns;
        consume_json["consume_deps"] = consume.consume_deps;
        consume_json["consume_start_deps"] = consume.consume_start_deps;
        consume_json["gate_id"] = consume.gate_id;
        consume_json["compute_share"] = consume.compute_share
            ? projection_share_wire(*consume.compute_share)
            : Json(nullptr);
        consume_json["hbm_share"] = consume.hbm_share
            ? projection_share_wire(*consume.hbm_share)
            : Json(nullptr);
        chain_json["consume"] = std::move(consume_json);
        chains_json.push_back(std::move(chain_json));
    }
    Json out;
    out["version"] = version;
    out["chains"] = std::move(chains_json);
    return out;
}

// _anchor_wire.
Json anchor_wire(const Anchor& anchor) {
    Json selector;
    selector["kind"] = anchor.selector.kind;
    if (anchor.selector.group_ordinal) {
        selector["group_ordinal"] = *anchor.selector.group_ordinal;
    }
    if (anchor.selector.tile_ordinal) {
        selector["tile_ordinal"] = *anchor.selector.tile_ordinal;
    }
    if (anchor.selector.shard_ordinal) {
        selector["shard_ordinal"] = *anchor.selector.shard_ordinal;
    }
    if (anchor.selector.endpoint) {
        selector["endpoint"] = *anchor.selector.endpoint;
    }
    if (anchor.selector.existing_gate) {
        selector["existing_gate"] = *anchor.selector.existing_gate;
    }
    if (anchor.selector.tag_allocation_emission_ordinal) {
        selector["tag_allocation_emission_ordinal"] =
            *anchor.selector.tag_allocation_emission_ordinal;
    }
    if (anchor.selector.tag_offset) {
        selector["tag_offset"] = *anchor.selector.tag_offset;
    }
    Json out;
    out["role"] = anchor.role;
    out["rank"] = anchor.rank;
    out["binding"] = anchor.binding;
    out["node_id"] =
        anchor.node_id ? Json(*anchor.node_id) : Json(nullptr);
    out["selector"] = std::move(selector);
    return out;
}

// _tile_shard_data_bytes.
u64 tile_shard_data_bytes(const PipelineTileAxis& axis, u64 units, u64 offset,
                          const PipelineShardRule& shard) {
    if (axis.unit == "source_tokens") {
        return checked_mul(units, shard.bytes_per_unit,
                           "source-token shard bytes");
    }
    const u64 end = checked_add(offset, units, "stream-byte tile end");
    const u64 capped = std::min(shard.total_data_bytes, end);
    return capped > offset ? capped - offset : 0;
}

// _make_pipeline_shard_tile.
PipelineShardTile make_shard_tile(const PipelineShardRule& shard,
                                  u64 data_bytes) {
    return PipelineShardTile{shard.shard_ordinal, shard.source_rank,
                             shard.target_rank, data_bytes, data_bytes > 0};
}

}  // namespace

// ------------------------------------------------------------ from_wire --

DescriptorPtr PipelineDescriptor::from_wire(const Json& wire) {
    const Json& top = mapping(wire, "descriptor",
        {"descriptor_version", "identity", "stream_kind", "groups", "protocol",
         "prediction_profile", "execution_profile", "logical_id_ranges",
         "tag_allocation_runs", "endpoint_runs", "endpoint_dependency_runs",
         "consume_completion_runs", "projection", "anchors"});
    const u64 version =
        p_uint(top["descriptor_version"], "descriptor_version", kU32Max);
    if (version != 1) {
        fail("unsupported descriptor_version " + std::to_string(version));
    }

    // identity (canonical segment/domain encoding).
    const Json& identity_raw = mapping(top["identity"], "identity",
        {"request_id", "stage", "generation", "stage_epoch", "segment_ordinal",
         "rank", "staging_allocation_id", "segment_id", "buffer_domain_id"});
    const std::string request =
        p_string(identity_raw["request_id"], "identity.request_id");
    const std::string stage =
        p_enum(identity_raw["stage"], "identity.stage", {"prefill", "decode"});
    const u64 generation =
        p_uint(identity_raw["generation"], "identity.generation");
    const u64 stage_epoch =
        p_uint(identity_raw["stage_epoch"], "identity.stage_epoch");
    const u64 segment_ordinal =
        p_uint(identity_raw["segment_ordinal"], "identity.segment_ordinal");
    const u64 rank = p_rank(identity_raw["rank"], "identity.rank");
    const u64 allocation = p_uint(identity_raw["staging_allocation_id"],
                                  "identity.staging_allocation_id");
    const std::string segment_id =
        p_string(identity_raw["segment_id"], "identity.segment_id");
    const std::string domain_id =
        p_string(identity_raw["buffer_domain_id"], "identity.buffer_domain_id");
    const std::string request_hex = hex_encode(request);
    const std::string expected_segment =
        "pp1/" + request_hex + "/" + stage + "/" + std::to_string(generation)
        + "/" + std::to_string(stage_epoch) + "/" + std::to_string(rank) + "/"
        + std::to_string(segment_ordinal);
    const std::string expected_domain =
        "pb1/" + request_hex + "/" + std::to_string(rank) + "/"
        + std::to_string(allocation);
    if (segment_id != expected_segment || domain_id != expected_domain) {
        fail("identity IDs do not match canonical encoding");
    }

    const std::string stream_kind = p_enum(
        top["stream_kind"], "stream_kind",
        {"prefill_read_window", "decode_remote_credit"});
    std::vector<PipelineGroup> groups;
    {
        const Json& groups_raw = p_array(top["groups"], "groups", true);
        groups.reserve(groups_raw.size());
        for (std::size_t i = 0; i < groups_raw.size(); ++i) {
            groups.push_back(
                parse_group(groups_raw[i], at_index("groups", i)));
        }
        for (std::size_t i = 1; i < groups.size(); ++i) {
            if (groups[i - 1].group_ordinal >= groups[i].group_ordinal) {
                fail("groups must preserve strictly increasing group_ordinal");
            }
        }
        const char* expected_stage_kind =
            stage == "prefill" ? "prefill_read_window" : "decode_remote_credit";
        if (stream_kind != expected_stage_kind) {
            fail("stream_kind does not match identity.stage");
        }
    }

    const PipelineProtocol protocol = parse_protocol(top["protocol"]);
    const Json prediction_wire = parse_prediction_profile(
        top["prediction_profile"]);
    const Json execution_wire = parse_execution_profile(
        top["execution_profile"]);

    std::vector<IdRange> id_ranges;
    {
        const Json& ranges_raw = p_array(top["logical_id_ranges"],
                                         "logical_id_ranges");
        id_ranges.reserve(ranges_raw.size());
        for (std::size_t i = 0; i < ranges_raw.size(); ++i) {
            id_ranges.push_back(
                parse_id_range(ranges_raw[i], at_index("logical_id_ranges", i)));
        }
        validate_id_ranges(id_ranges);
    }

    std::vector<TagRun> tag_runs;
    {
        const Json& runs_raw = p_array(top["tag_allocation_runs"],
                                       "tag_allocation_runs");
        tag_runs.reserve(runs_raw.size());
        for (std::size_t i = 0; i < runs_raw.size(); ++i) {
            tag_runs.push_back(
                parse_tag_run(runs_raw[i], at_index("tag_allocation_runs", i)));
        }
        validate_tag_runs(tag_runs, groups);
    }

    std::vector<PipelineEndpointRun> endpoint_runs;
    {
        const Json& runs_raw = p_array(top["endpoint_runs"], "endpoint_runs");
        endpoint_runs.reserve(runs_raw.size());
        for (std::size_t i = 0; i < runs_raw.size(); ++i) {
            endpoint_runs.push_back(parse_endpoint_run(
                runs_raw[i], at_index("endpoint_runs", i)));
        }
        validate_endpoint_runs(endpoint_runs, groups, tag_runs);
    }

    std::vector<DependencyRun> dependency_runs;
    {
        const Json& runs_raw = p_array(top["endpoint_dependency_runs"],
                                       "endpoint_dependency_runs");
        dependency_runs.reserve(runs_raw.size());
        for (std::size_t i = 0; i < runs_raw.size(); ++i) {
            dependency_runs.push_back(parse_affine_dependency_run(
                runs_raw[i], at_index("endpoint_dependency_runs", i), true));
        }
    }
    std::vector<PipelineConsumeCompletionRun> completion_runs;
    {
        const Json& runs_raw = p_array(top["consume_completion_runs"],
                                       "consume_completion_runs");
        completion_runs.reserve(runs_raw.size());
        for (std::size_t i = 0; i < runs_raw.size(); ++i) {
            const DependencyRun parsed = parse_affine_dependency_run(
                runs_raw[i], at_index("consume_completion_runs", i), false);
            PipelineConsumeCompletionRun run;
            run.rank = parsed.rank;
            run.group_ordinal = parsed.group_ordinal;
            run.first_tile = parsed.first_tile;
            run.tile_count = parsed.tile_count;
            run.shard_ordinals = parsed.shard_ordinals;
            run.dependency_kind = parsed.dependency_kind;
            run.first_dependency_json_id = parsed.first_dependency_json_id;
            run.dependency_tile_stride = parsed.dependency_tile_stride;
            run.dependency_shard_stride = parsed.dependency_shard_stride;
            completion_runs.push_back(std::move(run));
        }
    }
    validate_endpoint_dependency_runs(dependency_runs, endpoint_runs, groups);
    if (!endpoint_runs.empty()) {
        validate_consume_completion_runs(completion_runs, groups, rank);
    } else if (!completion_runs.empty()) {
        fail("projection-only descriptor must not include consume completion"
             " runs");
    }

    // projection (parse + validate).
    u64 projection_version = 0;
    std::vector<ProjectionChain> chains;
    {
        const Json& raw = mapping(top["projection"], "projection",
                                  {"version", "chains"});
        projection_version = p_uint(raw["version"], "projection.version",
                                    kU32Max);
        if (projection_version != 1) {
            fail("unsupported projection version "
                 + std::to_string(projection_version));
        }
        const Json& chains_raw = p_array(raw["chains"], "projection.chains");
        chains.reserve(chains_raw.size());
        for (std::size_t i = 0; i < chains_raw.size(); ++i) {
            const std::string path = at_index("projection.chains", i);
            const Json& chain_raw = mapping(chains_raw[i], path,
                {"group_ordinal", "rank", "tile_count", "fill", "consume"});
            ProjectionChain chain;
            chain.tile_count =
                p_positive(chain_raw["tile_count"], path + ".tile_count");
            chain.group_ordinal =
                p_uint(chain_raw["group_ordinal"], path + ".group_ordinal");
            chain.rank = p_rank(chain_raw["rank"], path + ".rank");
            chain.fill = parse_projection_fill(chain_raw["fill"], path + ".fill",
                                               chain.tile_count, protocol);
            chain.consume = parse_projection_consume(
                chain_raw["consume"], path + ".consume", chain.tile_count);
            chains.push_back(std::move(chain));
        }
        validate_projection(chains, groups, protocol, rank);
    }

    std::vector<Anchor> anchors;
    {
        const Json& anchors_raw = p_array(top["anchors"], "anchors");
        anchors.reserve(anchors_raw.size());
        for (std::size_t i = 0; i < anchors_raw.size(); ++i) {
            anchors.push_back(parse_anchor(anchors_raw[i], at_index("anchors", i)));
        }
    }
    const bool execution_descriptor =
        !id_ranges.empty() || !tag_runs.empty() || !endpoint_runs.empty();
    if (execution_descriptor) {
        if (anchors.empty()) {
            fail("execution descriptor needs endpoint/service anchors");
        }
        validate_anchors(anchors, groups, tag_runs, endpoint_runs);
        validate_endpoint_id_targets(endpoint_runs, id_ranges, anchors);
        bool has_start = false;
        bool has_frontier = false;
        bool has_service = false;
        bool has_settle = false;
        for (const Anchor& anchor : anchors) {
            has_start = has_start || anchor.role == "start";
            has_frontier = has_frontier || anchor.role == "frontier";
            has_service = has_service || anchor.role == "service";
            has_settle = has_settle || anchor.role == "settle";
        }
        if (!(has_start && has_frontier && has_service && has_settle)) {
            fail("execution descriptor needs start/frontier/service/settle"
                 " anchors");
        }
    } else if (!anchors.empty()) {
        fail("projection-only descriptor must not invent execution anchors");
    }

    auto descriptor = std::make_shared<PipelineDescriptor>();
    descriptor->identity = PipelineIdentity{
        request, stage, generation, stage_epoch, segment_ordinal, rank,
        allocation, segment_id, domain_id};
    descriptor->stream_kind = stream_kind;
    descriptor->groups = std::move(groups);
    descriptor->protocol = protocol;
    descriptor->endpoint_runs = std::move(endpoint_runs);
    descriptor->consume_completion_runs = std::move(completion_runs);
    descriptor->descriptor_version = version;
    descriptor->prediction_profile_wire = prediction_wire;
    descriptor->execution_profile_wire = execution_wire;
    {
        Json ranges_json = Json::array();
        for (const IdRange& item : id_ranges) {
            Json entry;
            entry["rank"] = item.rank;
            entry["first_json_id"] = item.first;
            entry["count"] = item.count;
            entry["stride"] = item.stride;
            entry["kind"] = "virtual_endpoint_hole";
            ranges_json.push_back(std::move(entry));
        }
        descriptor->logical_id_ranges_wire = std::move(ranges_json);
    }
    {
        Json runs_json = Json::array();
        for (const TagRun& item : tag_runs) {
            Json entry;
            entry["allocator_scope_id"] = item.allocator_scope_id;
            entry["emission_ordinal"] = item.emission_ordinal;
            entry["first_tag"] = item.first_tag;
            entry["count"] = item.count;
            entry["group_ordinal"] = item.group_ordinal;
            entry["first_tile"] = item.first_tile;
            entry["first_shard"] = item.first_shard;
            entry["tile_count"] = item.tile_count;
            entry["shard_count"] = item.shard_count;
            runs_json.push_back(std::move(entry));
        }
        descriptor->tag_allocation_runs_wire = std::move(runs_json);
    }
    {
        Json runs_json = Json::array();
        for (const DependencyRun& item : dependency_runs) {
            runs_json.push_back(dependency_run_wire(item));
        }
        descriptor->endpoint_dependency_runs_wire = std::move(runs_json);
    }
    descriptor->projection_wire =
        projection_to_wire(projection_version, chains);
    {
        Json anchors_json = Json::array();
        for (const Anchor& anchor : anchors) {
            anchors_json.push_back(anchor_wire(anchor));
        }
        descriptor->anchors_wire = std::move(anchors_json);
    }
    return descriptor;
}

// ------------------------------------------------------------- to_wire --

Json PipelineDescriptor::to_wire() const {
    Json out;
    out["descriptor_version"] = descriptor_version;
    Json identity_json;
    identity_json["request_id"] = identity.request_id;
    identity_json["stage"] = identity.stage;
    identity_json["generation"] = identity.generation;
    identity_json["stage_epoch"] = identity.stage_epoch;
    identity_json["segment_ordinal"] = identity.segment_ordinal;
    identity_json["rank"] = identity.rank;
    identity_json["staging_allocation_id"] = identity.staging_allocation_id;
    identity_json["segment_id"] = identity.segment_id;
    identity_json["buffer_domain_id"] = identity.buffer_domain_id;
    out["identity"] = std::move(identity_json);
    out["stream_kind"] = stream_kind;
    {
        Json groups_json = Json::array();
        for (const PipelineGroup& group : groups) {
            groups_json.push_back(group_wire(group));
        }
        out["groups"] = std::move(groups_json);
    }
    {
        Json protocol_json;
        protocol_json["buffer_slots"] = protocol.buffer_slots;
        protocol_json["release_lag_tiles"] = protocol.release_lag_tiles;
        protocol_json["initial_tiles_without_release"] =
            protocol.initial_tiles_without_release;
        protocol_json["fill_ack_bytes"] = protocol.fill_ack_bytes;
        protocol_json["release_ack_bytes"] = protocol.release_ack_bytes;
        protocol_json["hbm_charge"] = protocol.hbm_charge;
        protocol_json["target_rank_order"] = protocol.target_rank_order;
        protocol_json["source_rank_order"] = protocol.source_rank_order;
        protocol_json["tile_branching"] = protocol.tile_branching;
        out["protocol"] = std::move(protocol_json);
    }
    out["prediction_profile"] = prediction_profile_wire;
    out["execution_profile"] = execution_profile_wire;
    out["logical_id_ranges"] = logical_id_ranges_wire;
    out["tag_allocation_runs"] = tag_allocation_runs_wire;
    {
        Json runs_json = Json::array();
        for (const PipelineEndpointRun& run : endpoint_runs) {
            runs_json.push_back(endpoint_run_wire(run));
        }
        out["endpoint_runs"] = std::move(runs_json);
    }
    out["endpoint_dependency_runs"] = endpoint_dependency_runs_wire;
    {
        Json runs_json = Json::array();
        for (const PipelineConsumeCompletionRun& run : consume_completion_runs) {
            runs_json.push_back(completion_run_wire(run));
        }
        out["consume_completion_runs"] = std::move(runs_json);
    }
    out["projection"] = projection_wire;
    out["anchors"] = anchors_wire;
    return out;
}

// -------------------------------------------------------------- queries --

const PipelineGroup* PipelineDescriptor::group_by_ordinal(
        std::uint64_t group_ordinal) const {
    for (const PipelineGroup& group : groups) {
        if (group.group_ordinal == group_ordinal) return &group;
    }
    return nullptr;
}

std::uint64_t PipelineDescriptor::endpoint_id(std::size_t run_index,
                                              std::uint64_t tile_ordinal,
                                              std::uint64_t shard_ordinal,
                                              std::size_t leg_index) const {
    if (run_index >= endpoint_runs.size()) {
        fail("endpoint run ordinal is outside descriptor");
    }
    const PipelineEndpointRun& run = endpoint_runs[run_index];
    const i128 tile_end =
        static_cast<i128>(run.first_tile) + run.tile_count;
    if (static_cast<i128>(tile_ordinal) < static_cast<i128>(run.first_tile)
            || static_cast<i128>(tile_ordinal) >= tile_end) {
        fail("tile is outside endpoint run");
    }
    auto shard_it = std::find(run.shard_ordinals.begin(),
                              run.shard_ordinals.end(), shard_ordinal);
    if (shard_it == run.shard_ordinals.end()) {
        fail("shard is outside endpoint run");
    }
    const u64 local_shard = static_cast<u64>(
        shard_it - run.shard_ordinals.begin());
    if (leg_index >= run.legs.size()) {
        fail("leg is outside endpoint run");
    }
    const PipelineEndpointLegRun& leg = run.legs[leg_index];
    const u64 local_tile = tile_ordinal - run.first_tile;
    u64 value = checked_add(
        run.first_json_id,
        checked_mul(local_tile, run.tile_stride, "endpoint tile ID offset"),
        "endpoint ID");
    value = checked_add(
        value,
        checked_mul(local_shard, run.shard_stride, "endpoint shard ID offset"),
        "endpoint ID");
    return checked_add(value, leg.json_offset, "endpoint leg ID");
}

std::pair<std::uint64_t, PipelineShardTile>
PipelineDescriptor::selected_shard_tile(std::uint64_t group_ordinal,
                                        std::uint64_t tile_ordinal,
                                        std::uint64_t shard_ordinal) const {
    const PipelineGroup* group = group_by_ordinal(group_ordinal);
    if (group == nullptr) {
        fail("unknown group_ordinal " + std::to_string(group_ordinal));
    }
    const PipelineTileAxis& axis = group->tile_axis;
    if (tile_ordinal >= axis.tile_count) {
        fail("local tile " + std::to_string(tile_ordinal) + " outside group "
             + std::to_string(group->group_ordinal) + " count "
             + std::to_string(axis.tile_count));
    }
    if (shard_ordinal >= group->shards.size()) {
        fail("shard index " + std::to_string(shard_ordinal) + " outside group "
             + std::to_string(group->group_ordinal) + " count "
             + std::to_string(group->shards.size()));
    }
    const std::uint64_t global_ordinal =
        checked_add(axis.global_tile_first, tile_ordinal, "global tile ordinal");
    const std::uint64_t units =
        tile_ordinal + 1 == axis.tile_count ? axis.tail_units
                                            : axis.units_per_tile;
    const std::uint64_t offset =
        checked_mul(tile_ordinal, axis.units_per_tile, "tile unit offset");
    const PipelineShardRule& shard =
        group->shards[static_cast<std::size_t>(shard_ordinal)];
    const std::uint64_t data_bytes =
        tile_shard_data_bytes(axis, units, offset, shard);
    return {global_ordinal, make_shard_tile(shard, data_bytes)};
}

bool descriptor_wire_equal(const PipelineDescriptor& a,
                           const PipelineDescriptor& b) {
    // nlohmann operator== compares objects/arrays/scalars exactly like Python
    // dict/list equality over the to_wire forms (all-int/bool/string scalars
    // compare numerically only across int/float, never across bool).
    return a.to_wire() == b.to_wire();
}

}  // namespace jointledger
