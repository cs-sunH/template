/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
DiePlanPersistentCache.cc -- implementation of the opt-in cross-run
persistent DiePlan cache (W5 of the joint提速改造, 2026-10-17); see
DiePlanPersistentCache.hh for the semantics contract and
extern/die_model/src/Capi.cc (the C2 estimate cache) for the persistence
caliber this file mirrors.

File layout (little-endian): magic[8] "DIEPLNC1" | u32 fmt=1 | u32 fp_len
| fp bytes | u32 hwkey_len | hwkey bytes | u32 count | count * entry |
u64 fnv1a64 checksum over all preceding bytes.  Entry: u8 key_len(=16) |
key ascii | DieOpSpec | Hardware | DiePlanSummary | stages | eight f64
plan totals.  Entries are stored oldest-first; load replays them into the
LRU (the same discipline as the C2 estimate store).  Every field of the
public DieTypes.hh types round-trips bit-exactly through this codec
(doubles via their IEEE-754 bit patterns); the sole deliberate exception
is DiePlanData::flat_shape, which stays null on load and is rebuilt
lazily by the first Execution of the plan.

Serialization reads the plan exclusively through the public DiePlan
accessors; deserialization rebuilds a DiePlanData aggregate and freezes it
through the public DiePlan constructor.  After decoding an entry the
loader recomputes DieModel::plan_cache_key over the reconstructed (spec,
hardware) pair and requires it to equal the stored key -- a self-check
that binds every persisted plan to the library's own content addressing.
*/

#include "astra-sim/workload/DiePlanPersistentCache.hh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <utility>
#include <vector>

#include <unistd.h>

namespace AstraSim {
namespace {

// ---------------------------------------------------------------------------
// Constants and fingerprint

constexpr std::size_t kPlanCacheCap = 4096;
constexpr std::uint32_t kPlanCacheFmt = 1;
const char kPlanCacheMagic[8] = {'D', 'I', 'E', 'P', 'L', 'N', 'C', '1'};

// Generous structural sanity bounds for decoded vectors; a corrupt length
// that survives these still cannot overrun the cursor (take() bounds every
// read against the file size).
constexpr std::uint32_t kMaxSpans = 1u << 20;
constexpr std::uint32_t kMaxStages = 4096;
constexpr std::uint32_t kMaxDepsPerStage = 4096;

std::uint64_t fnv1a64(const char* data, std::size_t n) {
    std::uint64_t h = 14695981039346656037ULL;
    for (std::size_t i = 0; i < n; ++i) {
        h ^= static_cast<std::uint64_t>(static_cast<unsigned char>(data[i]));
        h *= 1099511628211ULL;
    }
    return h;
}

std::string fnv1a_16hex(const std::string& s) {
    const std::uint64_t h = fnv1a64(s.data(), s.size());
    static const char* kHex = "0123456789abcdef";
    std::string out(16, '0');
    std::uint64_t v = h;
    for (int i = 15; i >= 0; --i) {
        out[static_cast<std::size_t>(i)] = kHex[v & 0xF];
        v >>= 4;
    }
    return out;
}

// Build fingerprint rendered as 16 hex digits (same primitive as the plan
// keys).  Both stamps are injected by the repository root CMakeLists at
// configure time: extern/die_model/configs/VERSION content plus a
// compiler/build-type/flags digest and a per-source SHA256 fold over every
// translation unit that can change a compiled plan (the die_model mapping
// core, the adapter that keys it and this codec).  The format tag keeps
// the fold's口径 expressible: changing what enters the fingerprint bumps
// the tag and invalidates every persisted store whole-file.
std::string plan_cache_fingerprint() {
    std::string s = "astra-die-plan-cache-fp-v1|";
    s += DIE_PLAN_CACHE_VERSION_STAMP;
    s += "|";
    s += DIE_PLAN_CACHE_BUILD_STAMP;
    return fnv1a_16hex(s);
}

// ---------------------------------------------------------------------------
// Little-endian binary helpers (exact, byte-addressed, locale-free)

void put_u32le(std::string& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<char>((v >> (8 * i)) & 0xFFu));
    }
}

void put_u64le(std::string& out, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<char>((v >> (8 * i)) & 0xFFu));
    }
}

void put_f64le(std::string& out, double d) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &d, sizeof bits);  // exact IEEE-754 bit pattern
    put_u64le(out, bits);
}

void put_str(std::string& out, const std::string& s) {
    put_u32le(out, static_cast<std::uint32_t>(s.size()));
    out.append(s);
}

struct Cursor {
    const std::string& buf;
    std::size_t pos = 0;
    bool ok = true;

    void take(std::size_t n, std::string* out) {
        if (!ok || pos + n > buf.size() || pos + n < pos) {
            ok = false;
            return;
        }
        if (out != nullptr) {
            out->assign(buf, pos, n);
        }
        pos += n;
    }

    std::size_t remaining() const {
        return ok && pos <= buf.size() ? buf.size() - pos : 0;
    }

    std::uint8_t u8() {
        if (!ok || pos + 1 > buf.size()) {
            ok = false;
            return 0;
        }
        return static_cast<std::uint8_t>(static_cast<unsigned char>(buf[pos++]));
    }

    std::uint32_t u32() {
        if (!ok || pos + 4 > buf.size()) {
            ok = false;
            return 0;
        }
        std::uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            v |= static_cast<std::uint32_t>(
                     static_cast<unsigned char>(buf[pos + i]))
                 << (8 * i);
        }
        pos += 4;
        return v;
    }

    std::uint64_t u64() {
        if (!ok || pos + 8 > buf.size()) {
            ok = false;
            return 0;
        }
        std::uint64_t v = 0;
        for (int i = 0; i < 8; ++i) {
            v |= static_cast<std::uint64_t>(
                     static_cast<unsigned char>(buf[pos + i]))
                 << (8 * i);
        }
        pos += 8;
        return v;
    }

    double f64() {
        const std::uint64_t bits = u64();
        double d = 0.0;
        std::memcpy(&d, &bits, sizeof d);  // bit-exact (incl. -0.0/NaN)
        return d;
    }

    bool str(std::string* out) {
        const std::uint32_t n = u32();
        if (!ok) {
            return false;
        }
        take(n, out);
        return ok;
    }
};

// ---------------------------------------------------------------------------
// Enum codecs (u8 index over the DieTypes.hh declaration order; anything
// outside the declared count is a decode anomaly)

bool enum_fits(std::uint32_t v, std::size_t count) { return v < count; }

std::uint8_t encode_bool(bool v) { return v ? 1 : 0; }

bool decode_bool(std::uint8_t v) { return v != 0; }

// ---------------------------------------------------------------------------
// Entry encode (public accessors only)

void append_spec(std::string& out, const DieModel::DieOpSpec& s) {
    put_u64le(out, s.schema_version);
    out.push_back(static_cast<char>(static_cast<std::uint8_t>(s.op_kind)));
    out.push_back(static_cast<char>(static_cast<std::uint8_t>(s.phase)));
    out.push_back(static_cast<char>(static_cast<std::uint8_t>(s.dtype)));
    const DieModel::DieModelDims& md = s.model_dims;
    put_u64le(out, md.hidden);
    put_u64le(out, md.ffn);
    put_u64le(out, md.vocab);
    out.push_back(
        static_cast<char>(static_cast<std::uint8_t>(md.mlp_variant)));
    out.push_back(
        static_cast<char>(static_cast<std::uint8_t>(md.norm_type)));
    out.push_back(
        static_cast<char>(static_cast<std::uint8_t>(md.position_encoding)));
    put_u64le(out, md.num_q_heads);
    put_u64le(out, md.num_kv_heads);
    put_u64le(out, md.head_dim);
    const DieModel::DieTpSlice& tp = s.tp_slice;
    put_u64le(out, tp.tensor_parallel);
    put_u64le(out, tp.tensor_parallel_rank);
    put_u64le(out, tp.q_heads_local);
    put_u64le(out, tp.kv_heads_local);
    put_u64le(out, tp.ffn_local);
    put_u64le(out, tp.vocab_local);
    put_u32le(out, static_cast<std::uint32_t>(s.spans.size()));
    for (const DieModel::DieSpan& sp : s.spans) {
        put_str(out, sp.request_id);
        put_u64le(out, sp.q_tokens);
        put_u64le(out, sp.kv_tokens);
        put_u64le(out, sp.q_start_pos);
        put_u64le(out, sp.kv_start_pos);
    }
    put_u64le(out, s.layer_count);
    put_u64le(out, s.weight_passes);
    out.push_back(
        static_cast<char>(static_cast<std::uint8_t>(s.kv_access.identity)));
    out.push_back(
        static_cast<char>(static_cast<std::uint8_t>(s.fusion_policy)));
    put_str(out, s.provenance);
    put_str(out, s.config_hash);
}

void append_core_table_entry(std::string& out,
                             const DieModel::CoreTableEntry& e) {
    // [merge seam 2026-10-18] CoreTableEntry moved to the coarse_die_v2
    // shape: the CoreType enum column became noc_width_bits and the
    // GFLOP/s column became the INT4 GOPS compatibility scalar.  The
    // serialized-shape change is covered by the build fingerprint (this
    // translation unit is SHA256-folded into DIE_PLAN_CACHE_BUILD_STAMP, so
    // every pre-merge store fails the header digest and is discarded).
    put_f64le(out, e.sram_mb_per_core);
    put_u64le(out, e.noc_width_bits);
    put_f64le(out, e.vec_g_eqop_per_core);
    put_f64le(out, e.pe_int4_gops_per_core);
    put_f64le(out, e.core_side_ref_um);
}

void append_hardware(std::string& out, const DieModel::Hardware& h) {
    out.push_back(static_cast<char>(static_cast<std::uint8_t>(h.profile)));
    // [merge seam 2026-10-18] Hardware::core_type (CoreType enum) was
    // replaced by the string-backed legacy six-key marker; serialize the
    // flag + string so legacy and DOCX-form hardware still hash to distinct
    // cache keys.
    out.push_back(static_cast<char>(static_cast<std::uint8_t>(
        h.legacy_core_type_form ? 1 : 0)));
    put_str(out, h.legacy_core_type);
    put_f64le(out, h.sram_mb_per_core);
    put_f64le(out, h.compute_area_mm2);
    put_u64le(out, h.hbm_per_side);
    put_u64le(out, h.hbm_sides);
    put_u64le(out, h.noc_width_bits);
    append_core_table_entry(out, h.core_row);
    put_f64le(out, h.acore_mm2);
    put_u64le(out, h.mesh_n);
    put_u64le(out, h.cores);
    put_f64le(out, h.pe_peak_die);
    put_f64le(out, h.vec_peak_die);
    put_f64le(out, h.sram_usable_bytes_per_core);
    put_f64le(out, h.link_peak_bytes);
    put_u64le(out, h.hbm_count);
    put_f64le(out, h.hbm_capacity_bytes);
    put_f64le(out, h.hbm_bandwidth_bytes);
    put_f64le(out, h.eta_pe);
    put_f64le(out, h.eta_vec);
    put_f64le(out, h.eta_matrix_mem);
    put_f64le(out, h.eta_vector_mem);
    put_f64le(out, h.eta_noc);
    put_f64le(out, h.launch_us_per_macro_group);
    put_f64le(out, h.tx8_matrix_peak_flop_s);
    put_f64le(out, h.tx8_vector_peak_eqop_s);
    put_f64le(out, h.tx8_bw_matrix_bytes_s);
    put_f64le(out, h.tx8_bw_vector_bytes_s);
    put_f64le(out, h.tx8_extra_launch_s);
    put_f64le(out, h.tx8_extra_memory_latency_s);
    out.push_back(static_cast<char>(encode_bool(h.full_score_materialization)));
    out.push_back(static_cast<char>(encode_bool(h.skip_fully_masked_tiles)));
}

void append_summary(std::string& out, const DieModel::DiePlanSummary& s) {
    out.push_back(static_cast<char>(static_cast<std::uint8_t>(s.op_kind)));
    out.push_back(static_cast<char>(static_cast<std::uint8_t>(s.profile)));
    put_str(out, s.loop_kind);
    put_u64le(out, s.m_tile);
    put_u64le(out, s.n_tile);
    put_u64le(out, s.k_tile);
    put_u64le(out, s.group_rows);
    put_u64le(out, s.group_cols);
    put_u64le(out, s.contexts);
    put_u64le(out, s.k_parallel);
    out.push_back(static_cast<char>(encode_bool(s.weight_resident)));
    out.push_back(static_cast<char>(encode_bool(s.double_buffered)));
    out.push_back(static_cast<char>(encode_bool(s.fused)));
    out.push_back(static_cast<char>(encode_bool(s.fallback)));
    out.push_back(static_cast<char>(encode_bool(s.streaming_two_pass)));
    put_f64le(out, s.sram_peak_per_core);
    put_u64le(out, s.layer_count);
    put_u64le(out, s.weight_passes);
    put_u64le(out, s.launch_instances);
    put_u64le(out, s.waves);
    put_f64le(out, s.weight_hbm_bytes);
}

void append_stage(std::string& out, const DieModel::DieStage& s) {
    put_str(out, s.label);
    out.push_back(static_cast<char>(static_cast<std::uint8_t>(s.resource)));
    put_u64le(out, s.macro_group);
    put_u64le(out, s.wave_repeat);
    put_f64le(out, s.pe_core_flop);
    put_f64le(out, s.vec_core_eqop);
    put_f64le(out, s.noc_edge_bytes);
    put_f64le(out, s.timer_seconds);
    put_f64le(out, s.mem.actual_bytes);
    put_f64le(out, s.mem.service_units);
    put_f64le(out, s.mem.eta);
    out.push_back(static_cast<char>(encode_bool(s.mem.matrix_class)));
    put_u32le(out, static_cast<std::uint32_t>(s.deps.size()));
    for (const DieModel::DieStageDep& d : s.deps) {
        put_u32le(out, d.stage);
        put_u32le(out, d.lag);
    }
}

// One entry: key + every serialized field of the plan (public accessors
// only).  flat_shape is intentionally absent (see file comment).
void append_entry(std::string& out, const std::string& key,
                  const DieModel::DiePlan& plan) {
    out.push_back(static_cast<char>(key.size()));
    out.append(key);
    append_spec(out, plan.spec());
    append_hardware(out, plan.hardware());
    append_summary(out, plan.summary());
    const std::vector<DieModel::DieStage>& stages = plan.stages();
    put_u32le(out, static_cast<std::uint32_t>(stages.size()));
    for (const DieModel::DieStage& stage : stages) {
        append_stage(out, stage);
    }
    put_f64le(out, plan.flops());
    put_f64le(out, plan.eqops());
    put_f64le(out, plan.matrix_bytes());
    put_f64le(out, plan.vector_bytes());
    put_f64le(out, plan.matrix_service_units());
    put_f64le(out, plan.vector_service_units());
    put_f64le(out, plan.noc_total_link_bytes());
    put_f64le(out, plan.noc_max_edge_bytes());
}

// ---------------------------------------------------------------------------
// Entry decode (fail-closed: any anomaly returns false / clears the cursor)

bool decode_core_table_entry(Cursor& c, DieModel::CoreTableEntry* e) {
    e->sram_mb_per_core = c.f64();
    e->noc_width_bits = c.u64();
    e->vec_g_eqop_per_core = c.f64();
    e->pe_int4_gops_per_core = c.f64();
    e->core_side_ref_um = c.f64();
    return c.ok;
}

bool decode_spec(Cursor& c, DieModel::DieOpSpec* s) {
    s->schema_version = c.u64();
    const std::uint8_t op_kind = c.u8();
    const std::uint8_t phase = c.u8();
    const std::uint8_t dtype = c.u8();
    if (!c.ok || !enum_fits(op_kind, 15) || !enum_fits(phase, 2) ||
        !enum_fits(dtype, 2)) {
        return false;
    }
    s->op_kind = static_cast<DieModel::OpKind>(op_kind);
    s->phase = static_cast<DieModel::Phase>(phase);
    s->dtype = static_cast<DieModel::Dtype>(dtype);
    DieModel::DieModelDims& md = s->model_dims;
    md.hidden = c.u64();
    md.ffn = c.u64();
    md.vocab = c.u64();
    const std::uint8_t mlp_variant = c.u8();
    const std::uint8_t norm_type = c.u8();
    const std::uint8_t position_encoding = c.u8();
    if (!c.ok || !enum_fits(mlp_variant, 2) || !enum_fits(norm_type, 2) ||
        !enum_fits(position_encoding, 2)) {
        return false;
    }
    md.mlp_variant = static_cast<DieModel::MlpVariant>(mlp_variant);
    md.norm_type = static_cast<DieModel::NormType>(norm_type);
    md.position_encoding =
        static_cast<DieModel::PositionEncoding>(position_encoding);
    md.num_q_heads = c.u64();
    md.num_kv_heads = c.u64();
    md.head_dim = c.u64();
    DieModel::DieTpSlice& tp = s->tp_slice;
    tp.tensor_parallel = c.u64();
    tp.tensor_parallel_rank = c.u64();
    tp.q_heads_local = c.u64();
    tp.kv_heads_local = c.u64();
    tp.ffn_local = c.u64();
    tp.vocab_local = c.u64();
    const std::uint32_t span_count = c.u32();
    if (!c.ok || span_count > kMaxSpans || span_count > c.remaining()) {
        return false;
    }
    s->spans.resize(span_count);
    for (DieModel::DieSpan& sp : s->spans) {
        if (!c.str(&sp.request_id)) {
            return false;
        }
        sp.q_tokens = c.u64();
        sp.kv_tokens = c.u64();
        sp.q_start_pos = c.u64();
        sp.kv_start_pos = c.u64();
        if (!c.ok) {
            return false;
        }
    }
    s->layer_count = c.u64();
    s->weight_passes = c.u64();
    const std::uint8_t kv_identity = c.u8();
    const std::uint8_t fusion_policy = c.u8();
    if (!c.ok || !enum_fits(kv_identity, 3) || !enum_fits(fusion_policy, 2)) {
        return false;
    }
    s->kv_access.identity = static_cast<DieModel::KvIdentity>(kv_identity);
    s->fusion_policy = static_cast<DieModel::FusionPolicy>(fusion_policy);
    return c.str(&s->provenance) && c.str(&s->config_hash) && c.ok;
}

bool decode_hardware(Cursor& c, DieModel::Hardware* h) {
    const std::uint8_t profile = c.u8();
    const std::uint8_t legacy_form = c.u8();
    if (!c.ok || !enum_fits(profile, 2) || legacy_form > 1) {
        return false;
    }
    h->profile = static_cast<DieModel::Profile>(profile);
    h->legacy_core_type_form = legacy_form != 0;
    if (!c.str(&h->legacy_core_type)) {
        return false;
    }
    h->sram_mb_per_core = c.f64();
    h->compute_area_mm2 = c.f64();
    h->hbm_per_side = c.u64();
    h->hbm_sides = c.u64();
    h->noc_width_bits = c.u64();
    if (!c.ok || !decode_core_table_entry(c, &h->core_row)) {
        return false;
    }
    h->acore_mm2 = c.f64();
    h->mesh_n = c.u64();
    h->cores = c.u64();
    h->pe_peak_die = c.f64();
    h->vec_peak_die = c.f64();
    h->sram_usable_bytes_per_core = c.f64();
    h->link_peak_bytes = c.f64();
    h->hbm_count = c.u64();
    h->hbm_capacity_bytes = c.f64();
    h->hbm_bandwidth_bytes = c.f64();
    h->eta_pe = c.f64();
    h->eta_vec = c.f64();
    h->eta_matrix_mem = c.f64();
    h->eta_vector_mem = c.f64();
    h->eta_noc = c.f64();
    h->launch_us_per_macro_group = c.f64();
    h->tx8_matrix_peak_flop_s = c.f64();
    h->tx8_vector_peak_eqop_s = c.f64();
    h->tx8_bw_matrix_bytes_s = c.f64();
    h->tx8_bw_vector_bytes_s = c.f64();
    h->tx8_extra_launch_s = c.f64();
    h->tx8_extra_memory_latency_s = c.f64();
    const std::uint8_t full_score = c.u8();
    const std::uint8_t skip_masked = c.u8();
    if (!c.ok) {
        return false;
    }
    h->full_score_materialization = decode_bool(full_score);
    h->skip_fully_masked_tiles = decode_bool(skip_masked);
    return true;
}

bool decode_summary(Cursor& c, DieModel::DiePlanSummary* s) {
    const std::uint8_t op_kind = c.u8();
    const std::uint8_t profile = c.u8();
    if (!c.ok || !enum_fits(op_kind, 15) || !enum_fits(profile, 2)) {
        return false;
    }
    s->op_kind = static_cast<DieModel::OpKind>(op_kind);
    s->profile = static_cast<DieModel::Profile>(profile);
    if (!c.str(&s->loop_kind)) {
        return false;
    }
    s->m_tile = c.u64();
    s->n_tile = c.u64();
    s->k_tile = c.u64();
    s->group_rows = c.u64();
    s->group_cols = c.u64();
    s->contexts = c.u64();
    s->k_parallel = c.u64();
    const std::uint8_t weight_resident = c.u8();
    const std::uint8_t double_buffered = c.u8();
    const std::uint8_t fused = c.u8();
    const std::uint8_t fallback = c.u8();
    const std::uint8_t streaming_two_pass = c.u8();
    if (!c.ok) {
        return false;
    }
    s->weight_resident = decode_bool(weight_resident);
    s->double_buffered = decode_bool(double_buffered);
    s->fused = decode_bool(fused);
    s->fallback = decode_bool(fallback);
    s->streaming_two_pass = decode_bool(streaming_two_pass);
    s->sram_peak_per_core = c.f64();
    s->layer_count = c.u64();
    s->weight_passes = c.u64();
    s->launch_instances = c.u64();
    s->waves = c.u64();
    s->weight_hbm_bytes = c.f64();
    return c.ok;
}

bool decode_stage(Cursor& c, DieModel::DieStage* s) {
    if (!c.str(&s->label)) {
        return false;
    }
    const std::uint8_t resource = c.u8();
    if (!c.ok || !enum_fits(resource, 6)) {
        return false;
    }
    s->resource = static_cast<DieModel::StageResource>(resource);
    s->macro_group = c.u64();
    s->wave_repeat = c.u64();
    s->pe_core_flop = c.f64();
    s->vec_core_eqop = c.f64();
    s->noc_edge_bytes = c.f64();
    s->timer_seconds = c.f64();
    s->mem.actual_bytes = c.f64();
    s->mem.service_units = c.f64();
    s->mem.eta = c.f64();
    s->mem.matrix_class = decode_bool(c.u8());
    if (!c.ok) {
        return false;
    }
    const std::uint32_t dep_count = c.u32();
    if (!c.ok || dep_count > kMaxDepsPerStage) {
        return false;
    }
    s->deps.resize(dep_count);
    for (DieModel::DieStageDep& d : s->deps) {
        d.stage = c.u32();
        d.lag = c.u32();
        if (!c.ok) {
            return false;
        }
    }
    return true;
}

struct DecodedEntry {
    std::string key;
    std::shared_ptr<const DieModel::DiePlan> plan;
};

// Decodes one entry and rebuilds the immutable plan.  The final gate
// recomputes DieModel::plan_cache_key over the reconstructed (spec,
// hardware) and requires it to equal the stored key, binding every loaded
// plan to the library's content addressing.  Returns false on ANY
// anomaly; the caller then discards the whole file.
bool decode_entry(Cursor& c, std::size_t stage_count_bound,
                  DecodedEntry* out) {
    const std::uint8_t key_len = c.u8();
    if (!c.ok || key_len != 16) {
        return false;
    }
    c.take(key_len, &out->key);
    if (!c.ok) {
        return false;
    }
    DieModel::DieOpSpec spec;
    if (!decode_spec(c, &spec)) {
        return false;
    }
    DieModel::Hardware hw;
    if (!decode_hardware(c, &hw)) {
        return false;
    }
    DieModel::DiePlanSummary summary;
    if (!decode_summary(c, &summary)) {
        return false;
    }
    const std::uint32_t stage_count = c.u32();
    if (!c.ok || stage_count > kMaxStages ||
        stage_count > stage_count_bound) {
        return false;
    }
    std::vector<DieModel::DieStage> stages(stage_count);
    for (std::uint32_t i = 0; i < stage_count; ++i) {
        if (!decode_stage(c, &stages[static_cast<std::size_t>(i)])) {
            return false;
        }
        for (const DieModel::DieStageDep& d :
             stages[static_cast<std::size_t>(i)].deps) {
            if (d.stage >= stage_count) {
                return false;  // dependency off the stage list
            }
        }
    }
    auto data = std::make_shared<DieModel::DiePlanData>();
    data->spec = std::make_shared<const DieModel::DieOpSpec>(
        std::move(spec));
    data->hardware = hw;
    data->summary = std::move(summary);
    data->stages = std::move(stages);
    data->flops = c.f64();
    data->eqops = c.f64();
    data->matrix_bytes = c.f64();
    data->vector_bytes = c.f64();
    data->matrix_service_units = c.f64();
    data->vector_service_units = c.f64();
    data->noc_total_link_bytes = c.f64();
    data->noc_max_edge_bytes = c.f64();
    if (!c.ok) {
        return false;
    }
    // flat_shape stays null: the first Execution of this plan rebuilds it
    // lazily through DiePlan::flat_shape_handle().
    out->plan = std::make_shared<const DieModel::DiePlan>(std::move(data));
    if (!out->plan->valid()) {
        return false;
    }
    if (DieModel::plan_cache_key(out->plan->spec(), out->plan->hardware()) !=
        out->key) {
        return false;  // content-addressing self-check failed
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Singleton lifecycle

DiePlanPersistentCache& DiePlanPersistentCache::instance() {
    static DiePlanPersistentCache cache;
    return cache;
}

bool DiePlanPersistentCache::enabled() {
    init_once();
    return enabled_;
}

void DiePlanPersistentCache::init_once() {
    if (initialized_) {
        return;
    }
    initialized_ = true;
    fingerprint_ = plan_cache_fingerprint();
    const char* path = std::getenv("DIE_PLAN_CACHE_PATH");
    if (path != nullptr && path[0] != '\0') {
        enabled_ = true;
        path_ = path;
    }
}

// Fail-closed load: ANY anomaly (absent file, bad magic/fmt, fingerprint
// or hardware-key mismatch, short/trailing bytes, checksum mismatch,
// decode failure, per-entry key self-check failure) leaves an empty store
// and never throws.
void DiePlanPersistentCache::load_file() {
    lru_.clear();
    index_.clear();
    dirty_ = false;
    try {
        std::ifstream in(path_, std::ios::binary);
        if (!in) {
            return;
        }
        std::string buf((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
        if (in.bad()) {
            return;
        }
        Cursor c{buf};
        std::string magic;
        c.take(8, &magic);
        if (!c.ok || magic != std::string(kPlanCacheMagic, 8)) return;
        if (c.u32() != kPlanCacheFmt) return;
        const std::uint32_t fp_len = c.u32();
        if (!c.ok || fp_len == 0 || fp_len > 4096) return;
        std::string fp;
        c.take(fp_len, &fp);
        if (!c.ok || fp != fingerprint_) return;  // stale build: drop all
        const std::uint32_t hw_len = c.u32();
        // The hardware key is the full "hw{...}" canonical summary (a few
        // hundred bytes of to_string doubles), so the sanity bound only
        // guards the allocation; take() still bounds it against the file.
        if (!c.ok || hw_len == 0 || hw_len > 4096) return;
        std::string stored_hw_key;
        c.take(hw_len, &stored_hw_key);
        if (!c.ok || stored_hw_key != hw_key_) {
            return;  // saved for another resolved hardware: drop all
        }
        const std::uint32_t count = c.u32();
        if (!c.ok || count > kPlanCacheCap) return;
        std::vector<DecodedEntry> entries;
        entries.reserve(count);
        for (std::uint32_t i = 0; i < count && c.ok; ++i) {
            DecodedEntry e;
            if (!decode_entry(c, c.remaining(), &e)) {
                return;  // whole file discard on any entry anomaly
            }
            entries.push_back(std::move(e));
        }
        const std::uint64_t stored_sum = c.u64();
        if (!c.ok || c.pos != buf.size()) return;  // short / trailing junk
        if (fnv1a64(buf.data(), c.pos - 8) != stored_sum) return;
        for (const DecodedEntry& e : entries) {
            // insert_loaded semantics: warming never marks the store
            // dirty, so a read-only run never rewrites the file.
            auto it = index_.find(e.key);
            if (it != index_.end()) {
                it->second->second = e.plan;
                lru_.splice(lru_.begin(), lru_, it->second);
                continue;
            }
            lru_.emplace_front(e.key, e.plan);
            index_[e.key] = lru_.begin();
            while (lru_.size() > kPlanCacheCap) {
                index_.erase(lru_.back().first);
                lru_.pop_back();
            }
        }
    } catch (...) {
        // Fail-closed: an unexpected allocation failure leaves the store
        // empty, never a partially loaded one.
        lru_.clear();
        index_.clear();
        dirty_ = false;
    }
}

// Best-effort save: temp file + atomic rename, swallows every failure so
// it can run from the noexcept clean-exit path.
void DiePlanPersistentCache::save_file() const noexcept {
    try {
        std::string entries;
        std::size_t cnt = 0;
        for (auto it = lru_.rbegin(); it != lru_.rend(); ++it) {
            append_entry(entries, it->first, *it->second);
            ++cnt;
        }
        std::string out;
        out.append(kPlanCacheMagic, 8);
        put_u32le(out, kPlanCacheFmt);
        put_u32le(out, static_cast<std::uint32_t>(fingerprint_.size()));
        out.append(fingerprint_);
        put_u32le(out, static_cast<std::uint32_t>(hw_key_.size()));
        out.append(hw_key_);
        put_u32le(out, static_cast<std::uint32_t>(cnt));
        out.append(entries);
        put_u64le(out, fnv1a64(out.data(), out.size()));
        const std::string tmp =
            path_ + ".tmp." + std::to_string(static_cast<long>(::getpid()));
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            if (!f) {
                return;
            }
            f.write(out.data(), static_cast<std::streamsize>(out.size()));
            if (!f) {
                f.close();
                std::remove(tmp.c_str());
                return;
            }
        }
        if (std::rename(tmp.c_str(), path_.c_str()) != 0) {
            std::remove(tmp.c_str());
        }
    } catch (...) {
        // Persistence is best effort; never propagate out of the clean
        // exit path.
    }
}

// ---------------------------------------------------------------------------
// Public API

std::shared_ptr<const DieModel::DiePlan> DiePlanPersistentCache::get(
    const std::string& key, const DieModel::Hardware& runtime_hw) {
    init_once();
    if (!enabled_) {
        return nullptr;
    }
    if (!loaded_) {
        loaded_ = true;
        hw_key_ = DieModel::plan_cache_hardware_key(runtime_hw);
        load_file();
    }
    auto it = index_.find(key);
    if (it == index_.end()) {
        return nullptr;
    }
    lru_.splice(lru_.begin(), lru_, it->second);
    return it->second->second;
}

void DiePlanPersistentCache::put(
    const std::string& key, std::shared_ptr<const DieModel::DiePlan> plan) {
    init_once();
    if (!enabled_ || plan == nullptr || !plan->valid()) {
        return;
    }
    try {
        if (hw_key_.empty()) {
            // Unreachable in the adapter flow (get always runs first and
            // seeds the file-level hardware key); kept so a direct put
            // still produces a loadable file.
            hw_key_ = DieModel::plan_cache_hardware_key(plan->hardware());
        }
        auto it = index_.find(key);
        if (it != index_.end()) {
            it->second->second = std::move(plan);
            lru_.splice(lru_.begin(), lru_, it->second);
        } else {
            lru_.emplace_front(key, std::move(plan));
            index_[key] = lru_.begin();
            while (lru_.size() > kPlanCacheCap) {
                index_.erase(lru_.back().first);
                lru_.pop_back();
            }
        }
        dirty_ = true;
    } catch (...) {
        // Best effort: the in-process plan cache still served the compile.
    }
}

void DiePlanPersistentCache::flush_clean_exit() noexcept {
    DiePlanPersistentCache& cache = instance();
    cache.init_once();
    if (!cache.enabled_ || !cache.dirty_) {
        return;
    }
    cache.save_file();
    // A successful or failed save alike ends the store's write interest
    // for this process; a failed save is simply lost (best effort by
    // contract), never retried and never fails the clean exit.
    cache.dirty_ = false;
}

void DiePlanPersistentCache::reset_for_test() {
    DiePlanPersistentCache& cache = instance();
    cache.initialized_ = false;
    cache.enabled_ = false;
    cache.loaded_ = false;
    cache.dirty_ = false;
    cache.path_.clear();
    cache.fingerprint_.clear();
    cache.hw_key_.clear();
    cache.lru_.clear();
    cache.index_.clear();
}

}  // namespace AstraSim
