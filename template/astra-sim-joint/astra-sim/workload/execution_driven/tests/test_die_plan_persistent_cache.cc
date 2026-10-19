/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
test_die_plan_persistent_cache.cc -- unit test of the AstraSim-side
persistent DiePlan cache (W5 of the joint提速改造, 2026-10-17; see
astra-sim/workload/DiePlanPersistentCache.hh).

Coverage groups:

  1. Round trip (mesh_die_v1): put three compiled plans (single-span
     prefill, multi-span prefill under the collapsed v3 key, attention
     prefill under the per-span v2 key), flush, then reload in a simulated
     fresh process -- every reloaded plan must be BITWISE identical to the
     compiled original: the full DieOpSpec (spans, provenance, config
     hash), the full Hardware (core table row, derived scalars, both
     profile coefficient blocks, both tx8 flags), the summary, every stage
     (label, resource, repeats, per-instance doubles, memory triple, deps)
     and the eight scalar totals.  The lazy FlatShape rebuild is exercised
     end-to-end: driving the reloaded plan through the full Execution
     action protocol (Timer advance / MemoryRequest service) must reach
     Done with EXACTLY the same elapsed_ns as the original plan.
  2. Round trip (tx8_reference_v1): the reference-profile plan and its
     hardware constants round trip in a store of their own.
  3. Persistence hygiene: a loaded (not dirty) store never rewrites the
     file; a later compile merges into the store (count grows, every key
     still hits); a miss key stays a miss.
  4. Corruption matrix (fail-closed whole-file discard): truncation at the
     tail, truncation mid-entry, trailing garbage, wrong magic, wrong
     fingerprint, wrong hardware key, wrong checksum, and an entry whose
     decoded (spec, hardware) no longer reproduces its stored
     plan_cache_key.  The magic/fingerprint/hardware-key/entry mutations
     are applied WITH the file checksum recomputed, so each reject
     exercises its targeted header/entry check rather than the checksum;
     a positive control (unmutated bytes + recomputed checksum) proves
     the checksum surgery itself does not cause rejects.  After a rejected
     load the corrupt file is left untouched and a fresh compile heals the
     store in place.
  5. Switch discipline: DIE_PLAN_CACHE_PATH unset and empty both mean
     fully disabled (null lookups, no-op puts, no file ever created); a
     set-but-absent file behaves as an empty store that a put+flush
     creates.

Build (in-tree): the repository root CMakeLists registers this file next
to NativeProjectionCApiTest as DiePlanPersistentCacheTest (sources: this
file + DiePlanPersistentCache.cc; links die_model and reuses the SAME
configure-time stamp variables as the AstraSim target).  Exit code 0 on
ALL PASS; style follows the repository fixture convention (no gtest).
*/

#include "DieModel.hh"
#include "astra-sim/workload/DiePlanPersistentCache.hh"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

using AstraSim::DiePlanPersistentCache;

namespace {

// ---------------------------------------------------------------------------
// Checker (repository fixture style, self-contained)

struct Checker {
    bool ok = true;
    void expect(bool cond, const std::string& what) {
        if (!cond) {
            ok = false;
            std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        }
    }
    void expect_uint(std::uint64_t actual, std::uint64_t expected,
                     const std::string& what) {
        if (actual != expected) {
            ok = false;
            std::fprintf(stderr, "FAIL: %s = %llu, expected %llu\n",
                         what.c_str(),
                         static_cast<unsigned long long>(actual),
                         static_cast<unsigned long long>(expected));
        }
    }
};

// ---------------------------------------------------------------------------
// Fixtures (minimal inline copies of the die_model test helpers, so this
// test depends only on the die_model public API)

// Coarse merge, 2026-10-18: the merged die_model hardware parser (taken
// from the coarse HEAD) requires explicit equal mesh-nx/mesh-ny and rejects
// unknown keys, so the retired "core-type" alias fixture now carries the
// explicit 23x23 grid the shared die_model test helpers use.
nlohmann::json die_node(double sram, double area, std::uint64_t per_side,
                        std::uint64_t sides, std::uint64_t width,
                        std::uint64_t nx = 23, std::uint64_t ny = 23) {
    nlohmann::json die;
    die["sram-mb-per-core"] = sram;
    die["mesh-nx"] = nx;
    die["mesh-ny"] = ny;
    die["compute-area-mm2"] = area;
    die["hbm-per-side"] = per_side;
    die["hbm-sides"] = sides;
    die["noc-width-bits"] = width;
    return die;
}

nlohmann::json wrapped(const nlohmann::json& die, const char* profile) {
    nlohmann::json j;
    j["schema_version"] = 1u;
    j["profile"] = profile;
    j["die"] = die;
    return j;
}

DieModel::Hardware hw_mesh() {
    return DieModel::make_hardware(
        wrapped(die_node(0.75, 800, 4, 1, 512), "mesh_die_v1"));
}

DieModel::Hardware hw_tx8() {
    return DieModel::make_hardware(
        wrapped(die_node(0.75, 800, 3, 2, 512), "tx8_reference_v1"));
}

DieModel::DieOpSpec spec_of(DieModel::OpKind kind, DieModel::Phase phase,
                            std::uint64_t q, std::uint64_t kv,
                            std::uint64_t q_start, std::uint64_t kv_start,
                            std::uint64_t layers = 1) {
    DieModel::DieOpSpec s;
    s.op_kind = kind;
    s.phase = phase;
    s.dtype = DieModel::Dtype::Bf16;
    s.model_dims = DieModel::DieModelDims{
        4096, 14336, 32000, DieModel::MlpVariant::Swiglu,
        DieModel::NormType::Rmsnorm, DieModel::PositionEncoding::Rope, 32, 8,
        128};
    s.tp_slice = DieModel::DieTpSlice{1, 0, 32, 8, 14336, 32000};
    DieModel::DieSpan sp;
    sp.request_id = "r0";
    sp.q_tokens = q;
    sp.kv_tokens = kv;
    sp.q_start_pos = q_start;
    sp.kv_start_pos = kv_start;
    s.spans.push_back(sp);
    s.layer_count = layers;
    s.weight_passes = 1;
    s.provenance = "w5-test";
    s.config_hash = "w5";
    return s;
}

DieModel::DieOpSpec add_span(DieModel::DieOpSpec s, const std::string& id,
                             std::uint64_t q, std::uint64_t kv,
                             std::uint64_t q_start, std::uint64_t kv_start) {
    DieModel::DieSpan sp;
    sp.request_id = id;
    sp.q_tokens = q;
    sp.kv_tokens = kv;
    sp.q_start_pos = q_start;
    sp.kv_start_pos = kv_start;
    s.spans.push_back(sp);
    return s;
}

// ---------------------------------------------------------------------------
// Bitwise comparators (plan_key_normalization_test.cc caliber, extended to
// the spec and the full hardware)

static_assert(sizeof(double) == 8, "bitwise double compare assumes IEEE754");

bool bits_equal(double x, double y) {
    std::uint64_t a = 0;
    std::uint64_t b = 0;
    std::memcpy(&a, &x, sizeof(a));
    std::memcpy(&b, &y, sizeof(b));
    return a == b;
}

bool span_equal(const DieModel::DieSpan& a, const DieModel::DieSpan& b) {
    return a.request_id == b.request_id && a.q_tokens == b.q_tokens &&
           a.kv_tokens == b.kv_tokens && a.q_start_pos == b.q_start_pos &&
           a.kv_start_pos == b.kv_start_pos;
}

bool spec_equal(const DieModel::DieOpSpec& a, const DieModel::DieOpSpec& b) {
    if (a.schema_version != b.schema_version || a.op_kind != b.op_kind ||
        a.phase != b.phase || a.dtype != b.dtype) {
        return false;
    }
    const DieModel::DieModelDims& ma = a.model_dims;
    const DieModel::DieModelDims& mb = b.model_dims;
    if (ma.hidden != mb.hidden || ma.ffn != mb.ffn || ma.vocab != mb.vocab ||
        ma.mlp_variant != mb.mlp_variant || ma.norm_type != mb.norm_type ||
        ma.position_encoding != mb.position_encoding ||
        ma.num_q_heads != mb.num_q_heads ||
        ma.num_kv_heads != mb.num_kv_heads || ma.head_dim != mb.head_dim) {
        return false;
    }
    const DieModel::DieTpSlice& ta = a.tp_slice;
    const DieModel::DieTpSlice& tb = b.tp_slice;
    if (ta.tensor_parallel != tb.tensor_parallel ||
        ta.tensor_parallel_rank != tb.tensor_parallel_rank ||
        ta.q_heads_local != tb.q_heads_local ||
        ta.kv_heads_local != tb.kv_heads_local ||
        ta.ffn_local != tb.ffn_local || ta.vocab_local != tb.vocab_local) {
        return false;
    }
    if (a.spans.size() != b.spans.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.spans.size(); ++i) {
        if (!span_equal(a.spans[i], b.spans[i])) {
            return false;
        }
    }
    if (a.layer_count != b.layer_count ||
        a.weight_passes != b.weight_passes ||
        a.kv_access.identity != b.kv_access.identity ||
        a.fusion_policy != b.fusion_policy ||
        a.provenance != b.provenance || a.config_hash != b.config_hash) {
        return false;
    }
    return true;
}

bool core_row_equal(const DieModel::CoreTableEntry& a,
                    const DieModel::CoreTableEntry& b) {
    return bits_equal(a.sram_mb_per_core, b.sram_mb_per_core) &&
           a.noc_width_bits == b.noc_width_bits &&
           bits_equal(a.vec_g_eqop_per_core, b.vec_g_eqop_per_core) &&
           bits_equal(a.pe_int4_gops_per_core, b.pe_int4_gops_per_core) &&
           bits_equal(a.core_side_ref_um, b.core_side_ref_um);
}

bool hardware_equal(const DieModel::Hardware& a, const DieModel::Hardware& b) {
    if (a.profile != b.profile ||
        a.legacy_core_type_form != b.legacy_core_type_form ||
        a.legacy_core_type != b.legacy_core_type ||
        a.explicit_die_dimensions != b.explicit_die_dimensions ||
        !bits_equal(a.die_width_mm, b.die_width_mm) ||
        !bits_equal(a.die_height_mm, b.die_height_mm) ||
        !bits_equal(a.sram_mb_per_core, b.sram_mb_per_core) ||
        !bits_equal(a.compute_area_mm2, b.compute_area_mm2) ||
        a.hbm_per_side != b.hbm_per_side || a.hbm_sides != b.hbm_sides ||
        a.noc_width_bits != b.noc_width_bits ||
        !core_row_equal(a.core_row, b.core_row)) {
        return false;
    }
    if (!bits_equal(a.acore_mm2, b.acore_mm2) || a.mesh_n != b.mesh_n ||
        a.cores != b.cores || !bits_equal(a.pe_peak_die, b.pe_peak_die) ||
        !bits_equal(a.vec_peak_die, b.vec_peak_die) ||
        !bits_equal(a.sram_usable_bytes_per_core,
                    b.sram_usable_bytes_per_core) ||
        !bits_equal(a.link_peak_bytes, b.link_peak_bytes) ||
        a.hbm_count != b.hbm_count ||
        !bits_equal(a.hbm_capacity_bytes, b.hbm_capacity_bytes) ||
        !bits_equal(a.hbm_bandwidth_bytes, b.hbm_bandwidth_bytes)) {
        return false;
    }
    if (!bits_equal(a.eta_pe, b.eta_pe) || !bits_equal(a.eta_vec, b.eta_vec) ||
        !bits_equal(a.eta_matrix_mem, b.eta_matrix_mem) ||
        !bits_equal(a.eta_vector_mem, b.eta_vector_mem) ||
        !bits_equal(a.eta_noc, b.eta_noc) ||
        !bits_equal(a.launch_us_per_macro_group,
                    b.launch_us_per_macro_group)) {
        return false;
    }
    if (!bits_equal(a.tx8_matrix_peak_flop_s, b.tx8_matrix_peak_flop_s) ||
        !bits_equal(a.tx8_vector_peak_eqop_s, b.tx8_vector_peak_eqop_s) ||
        !bits_equal(a.tx8_bw_matrix_bytes_s, b.tx8_bw_matrix_bytes_s) ||
        !bits_equal(a.tx8_bw_vector_bytes_s, b.tx8_bw_vector_bytes_s) ||
        !bits_equal(a.tx8_extra_launch_s, b.tx8_extra_launch_s) ||
        !bits_equal(a.tx8_extra_memory_latency_s,
                    b.tx8_extra_memory_latency_s) ||
        a.full_score_materialization != b.full_score_materialization ||
        a.skip_fully_masked_tiles != b.skip_fully_masked_tiles) {
        return false;
    }
    return true;
}

bool stage_equal(const DieModel::DieStage& a, const DieModel::DieStage& b) {
    if (a.label != b.label || a.resource != b.resource ||
        a.macro_group != b.macro_group || a.wave_repeat != b.wave_repeat) {
        return false;
    }
    if (!bits_equal(a.pe_core_flop, b.pe_core_flop) ||
        !bits_equal(a.vec_core_eqop, b.vec_core_eqop) ||
        !bits_equal(a.noc_edge_bytes, b.noc_edge_bytes) ||
        !bits_equal(a.timer_seconds, b.timer_seconds)) {
        return false;
    }
    if (!bits_equal(a.mem.actual_bytes, b.mem.actual_bytes) ||
        !bits_equal(a.mem.service_units, b.mem.service_units) ||
        !bits_equal(a.mem.eta, b.mem.eta) ||
        a.mem.matrix_class != b.mem.matrix_class) {
        return false;
    }
    if (a.deps.size() != b.deps.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.deps.size(); ++i) {
        if (a.deps[i].stage != b.deps[i].stage || a.deps[i].lag != b.deps[i].lag) {
            return false;
        }
    }
    return true;
}

bool summary_equal(const DieModel::DiePlanSummary& a,
                   const DieModel::DiePlanSummary& b) {
    return a.op_kind == b.op_kind && a.profile == b.profile &&
           a.loop_kind == b.loop_kind && a.m_tile == b.m_tile &&
           a.n_tile == b.n_tile && a.k_tile == b.k_tile &&
           a.group_rows == b.group_rows && a.group_cols == b.group_cols &&
           a.contexts == b.contexts && a.k_parallel == b.k_parallel &&
           a.weight_resident == b.weight_resident &&
           a.double_buffered == b.double_buffered && a.fused == b.fused &&
           a.fallback == b.fallback &&
           a.streaming_two_pass == b.streaming_two_pass &&
           bits_equal(a.sram_peak_per_core, b.sram_peak_per_core) &&
           a.layer_count == b.layer_count &&
           a.weight_passes == b.weight_passes &&
           a.launch_instances == b.launch_instances && a.waves == b.waves &&
           bits_equal(a.weight_hbm_bytes, b.weight_hbm_bytes);
}

void expect_identical_products(Checker& t, const DieModel::DiePlan& pa,
                               const DieModel::DiePlan& pb,
                               const std::string& what) {
    t.expect(spec_equal(pa.spec(), pb.spec()),
             what + ": spec fields identical (incl. spans/strings)");
    t.expect(hardware_equal(pa.hardware(), pb.hardware()),
             what + ": hardware fields identical (incl. core row/tx8)");
    t.expect(summary_equal(pa.summary(), pb.summary()),
             what + ": summary fields identical");
    const std::vector<DieModel::DieStage>& sa = pa.stages();
    const std::vector<DieModel::DieStage>& sb = pb.stages();
    t.expect(sa.size() == sb.size(), what + ": stage count identical");
    const std::size_t n = sa.size() < sb.size() ? sa.size() : sb.size();
    for (std::size_t i = 0; i < n; ++i) {
        t.expect(stage_equal(sa[i], sb[i]),
                 what + ": stage " + std::to_string(i) +
                     " identical (incl. deps)");
    }
    t.expect(bits_equal(pa.flops(), pb.flops()) &&
                 bits_equal(pa.eqops(), pb.eqops()) &&
                 bits_equal(pa.matrix_bytes(), pb.matrix_bytes()) &&
                 bits_equal(pa.vector_bytes(), pb.vector_bytes()) &&
                 bits_equal(pa.matrix_service_units(),
                            pb.matrix_service_units()) &&
                 bits_equal(pa.vector_service_units(),
                            pb.vector_service_units()) &&
                 bits_equal(pa.noc_total_link_bytes(),
                            pb.noc_total_link_bytes()) &&
                 bits_equal(pa.noc_max_edge_bytes(), pb.noc_max_edge_bytes()),
             what + ": eight scalar totals bitwise identical");
}

// ---------------------------------------------------------------------------
// Full-speed Execution driver (execution_test.cc caliber): proves the
// lazily rebuilt FlatShape of a reloaded plan schedules identically.

std::uint64_t service_ns(double units, double bytes_per_second) {
    if (!(units > 0.0)) {
        return 0;
    }
    const double ns = std::ceil(units * 1e9 / bytes_per_second);
    return ns < 1.0 ? 1u : static_cast<std::uint64_t>(ns);
}

std::uint64_t drive_full_speed(DieModel::Execution& ex, double rate,
                               const std::string& what, Checker& t) {
    std::uint64_t guard = 0;
    for (;;) {
        const DieModel::ExecutionAction a = ex.next_action();
        if (a.kind == DieModel::ExecutionAction::Kind::Done) {
            return a.ns;
        }
        if (a.kind == DieModel::ExecutionAction::Kind::Timer) {
            t.expect(a.ns >= 1u, what + ": zero-nanosecond timer action");
            ex.advance_time(a.ns);
        } else {
            const std::uint64_t ns = service_ns(a.service_units, rate);
            if (ns != 0) {
                ex.advance_time(ns);
            }
            ex.complete_memory(a.stage_id);
        }
        if (++guard > 10000000u) {
            t.expect(false, what + ": action protocol did not terminate");
            return 0;
        }
    }
}

// ---------------------------------------------------------------------------
// File helpers and checksum surgery

std::string fnv1a64_bytes(const char* data, std::size_t n) {
    std::uint64_t h = 14695981039346656037ULL;
    for (std::size_t i = 0; i < n; ++i) {
        h ^= static_cast<std::uint64_t>(static_cast<unsigned char>(data[i]));
        h *= 1099511628211ULL;
    }
    // LE u64, rendered as 8 raw bytes (the file's checksum representation).
    std::string out(8, '\0');
    for (int i = 0; i < 8; ++i) {
        out[static_cast<std::size_t>(i)] =
            static_cast<char>((h >> (8 * i)) & 0xFFu);
    }
    return out;
}

bool read_file(const std::string& path, std::string* out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    out->assign((std::istreambuf_iterator<char>(in)),
                std::istreambuf_iterator<char>());
    return !in.bad();
}

bool write_file(const std::string& path, const std::string& buf) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        return false;
    }
    f.write(buf.data(), static_cast<std::streamsize>(buf.size()));
    return static_cast<bool>(f);
}

// Recomputes the trailing fnv1a64 checksum in place (used to aim each
// corruption case at its targeted check instead of the checksum).
void fix_checksum(std::string* buf) {
    const std::string sum = fnv1a64_bytes(buf->data(), buf->size() - 8);
    buf->replace(buf->size() - 8, 8, sum);
}

std::uint32_t le_u32(const std::string& b, std::size_t off) {
    return static_cast<std::uint32_t>(
               static_cast<unsigned char>(b[off])) |
           (static_cast<std::uint32_t>(
                static_cast<unsigned char>(b[off + 1]))
            << 8) |
           (static_cast<std::uint32_t>(
                static_cast<unsigned char>(b[off + 2]))
            << 16) |
           (static_cast<std::uint32_t>(
                static_cast<unsigned char>(b[off + 3]))
            << 24);
}

struct HeaderLayout {
    std::size_t fp_off = 0;
    std::uint32_t fp_len = 0;
    std::size_t hw_off = 0;
    std::uint32_t hw_len = 0;
    std::size_t count_off = 0;
    std::size_t entries_off = 0;
    std::uint32_t count = 0;
};

bool parse_layout(const std::string& b, HeaderLayout* h) {
    if (b.size() < 24) {
        return false;
    }
    h->fp_off = 16;
    h->fp_len = le_u32(b, 12);
    h->hw_off = h->fp_off + h->fp_len + 4;
    if (h->hw_off + 4 > b.size()) {
        return false;
    }
    h->hw_len = le_u32(b, h->hw_off - 4);
    h->count_off = h->hw_off + h->hw_len;
    if (h->count_off + 4 > b.size()) {
        return false;
    }
    h->count = le_u32(b, h->count_off);
    h->entries_off = h->count_off + 4;
    return true;
}

std::string unique_path(const char* tag) {
    return std::string("/tmp/die_plan_cache_w5_") + tag + "_" +
           std::to_string(static_cast<long>(::getpid())) + ".cache";
}

// Simulates a fresh process for `path`: resets the singleton, points the
// env var at `path` and returns the instance primed by one lookup.
void fresh_process(const std::string& path, const DieModel::Hardware& hw) {
    DiePlanPersistentCache::reset_for_test();
    ::setenv("DIE_PLAN_CACHE_PATH", path.c_str(), 1);
    DiePlanPersistentCache::instance().get("warm-the-store", hw);
}

}  // namespace

int main() {
    Checker t;
    const DieModel::Hardware mesh = hw_mesh();

    // Reference plans: single-span prefill (v3 collapsed key), multi-span
    // prefill with moved positions (still v3 for a non-attention op) and
    // attention prefill (v2 per-span key).
    DieModel::DieOpSpec s_norm = spec_of(DieModel::OpKind::Norm,
                                         DieModel::Phase::Prefill, 1004,
                                         1024, 0, 0);
    DieModel::DieOpSpec s_mlp = add_span(
        spec_of(DieModel::OpKind::MlpDown, DieModel::Phase::Prefill, 600,
                1024, 0, 0),
        "r1", 404, 2048, 9600, 512);
    DieModel::DieOpSpec s_av = spec_of(DieModel::OpKind::Av,
                                       DieModel::Phase::Prefill, 1024,
                                       1024, 0, 0);
    DieModel::DieOpSpec s_res = spec_of(DieModel::OpKind::Residual,
                                        DieModel::Phase::Decode, 1, 1, 0,
                                        0, 3);
    const std::shared_ptr<const DieModel::DiePlan> p_norm =
        DieModel::compile(s_norm, mesh);
    const std::shared_ptr<const DieModel::DiePlan> p_mlp =
        DieModel::compile(s_mlp, mesh);
    const std::shared_ptr<const DieModel::DiePlan> p_av =
        DieModel::compile(s_av, mesh);
    const std::shared_ptr<const DieModel::DiePlan> p_res =
        DieModel::compile(s_res, mesh);
    const std::string k_norm = DieModel::plan_cache_key(s_norm, mesh);
    const std::string k_mlp = DieModel::plan_cache_key(s_mlp, mesh);
    const std::string k_av = DieModel::plan_cache_key(s_av, mesh);
    const std::string k_res = DieModel::plan_cache_key(s_res, mesh);
    t.expect(k_norm.size() == 16 && k_mlp.size() == 16 && k_av.size() == 16,
             "plan keys are 16 hex digits");
    t.expect(k_norm != k_mlp && k_norm != k_av && k_mlp != k_av,
             "the three reference plans have distinct keys");

    // -----------------------------------------------------------------
    // Group 1: mesh round trip
    const std::string path_mesh = unique_path("mesh");
    ::remove(path_mesh.c_str());
    {
        DiePlanPersistentCache::reset_for_test();
        ::setenv("DIE_PLAN_CACHE_PATH", path_mesh.c_str(), 1);
        DiePlanPersistentCache& cache = DiePlanPersistentCache::instance();
        t.expect(cache.enabled(), "opted-in store reports enabled");
        cache.put(k_norm, p_norm);
        cache.put(k_mlp, p_mlp);
        cache.put(k_av, p_av);
        DiePlanPersistentCache::flush_clean_exit();

        std::string buf;
        t.expect(read_file(path_mesh, &buf), "flush created the store file");
        HeaderLayout lay;
        t.expect(parse_layout(buf, &lay), "store header parses");
        t.expect(buf.compare(0, 8, "DIEPLNC1") == 0, "store magic");
        t.expect(le_u32(buf, 8) == 1u, "store format version 1");
        t.expect(lay.fp_len == 16, "fingerprint is 16 hex digits");
        t.expect(lay.hw_len ==
                     DieModel::plan_cache_hardware_key(mesh).size(),
                 "hardware key length matches plan_cache_hardware_key");
        t.expect(buf.compare(lay.hw_off, lay.hw_len,
                             DieModel::plan_cache_hardware_key(mesh)) == 0,
                 "file header carries the mesh hardware key");
        t.expect_uint(lay.count, 3, "store holds three entries");
        t.expect(fnv1a64_bytes(buf.data(), buf.size() - 8) ==
                     buf.substr(buf.size() - 8, 8),
                 "trailing checksum covers the whole file");
    }
    {
        fresh_process(path_mesh, mesh);
        std::shared_ptr<const DieModel::DiePlan> r_norm;
        std::shared_ptr<const DieModel::DiePlan> r_mlp;
        std::shared_ptr<const DieModel::DiePlan> r_av;
        DiePlanPersistentCache& cache = DiePlanPersistentCache::instance();
        r_norm = cache.get(k_norm, mesh);
        r_mlp = cache.get(k_mlp, mesh);
        r_av = cache.get(k_av, mesh);
        t.expect(r_norm != nullptr && r_mlp != nullptr && r_av != nullptr,
                 "reloaded store hits all three keys");
        if (r_norm != nullptr && r_mlp != nullptr && r_av != nullptr) {
            expect_identical_products(t, *p_norm, *r_norm,
                                      "round-trip norm plan");
            expect_identical_products(t, *p_mlp, *r_mlp,
                                      "round-trip multi-span plan");
            expect_identical_products(t, *p_av, *r_av,
                                      "round-trip attention plan");
            // Lazy FlatShape rebuild must reproduce the exact schedule.
            DieModel::Execution e_orig(p_norm);
            DieModel::Execution e_reload(r_norm);
            const std::uint64_t t_orig = drive_full_speed(
                e_orig, mesh.hbm_bandwidth_bytes, "original norm", t);
            const std::uint64_t t_reload = drive_full_speed(
                e_reload, mesh.hbm_bandwidth_bytes, "reloaded norm", t);
            t.expect_uint(t_reload, t_orig,
                          "reloaded plan reaches Done with the original "
                          "elapsed_ns (flat-shape rebuild is exact)");
        }
        t.expect(cache.get("0000000000000000", mesh) == nullptr,
                 "unknown key misses");
    }

    // -----------------------------------------------------------------
    // Group 2: tx8 round trip in its own store (its hardware key differs)
    const std::string path_tx8 = unique_path("tx8");
    ::remove(path_tx8.c_str());
    {
        const DieModel::Hardware tx8 = hw_tx8();
        DieModel::DieOpSpec s_ref = spec_of(DieModel::OpKind::Norm,
                                            DieModel::Phase::Prefill, 1024,
                                            1024, 0, 0);
        const std::shared_ptr<const DieModel::DiePlan> p_ref =
            DieModel::compile(s_ref, tx8);
        const std::string k_ref = DieModel::plan_cache_key(s_ref, tx8);
        DiePlanPersistentCache::reset_for_test();
        ::setenv("DIE_PLAN_CACHE_PATH", path_tx8.c_str(), 1);
        DiePlanPersistentCache::instance().put(k_ref, p_ref);
        DiePlanPersistentCache::flush_clean_exit();

        fresh_process(path_tx8, tx8);
        const std::shared_ptr<const DieModel::DiePlan> r_ref =
            DiePlanPersistentCache::instance().get(k_ref, tx8);
        t.expect(r_ref != nullptr, "tx8 store hits its key");
        if (r_ref != nullptr) {
            expect_identical_products(t, *p_ref, *r_ref,
                                      "round-trip tx8 reference plan");
        }
        // A mesh-hardware process must reject the tx8 store whole-file
        // (file-level hardware key check).
        fresh_process(path_tx8, mesh);
        t.expect(DiePlanPersistentCache::instance().get(k_ref, mesh) ==
                     nullptr,
                 "mesh process discards a tx8-hardware store");
    }

    // -----------------------------------------------------------------
    // Group 3: persistence hygiene (no rewrite when clean; merge on put)
    {
        std::string before;
        t.expect(read_file(path_mesh, &before), "mesh store readable");
        fresh_process(path_mesh, mesh);
        DiePlanPersistentCache& cache = DiePlanPersistentCache::instance();
        (void)cache.get(k_norm, mesh);  // loads; must NOT dirty the store
        DiePlanPersistentCache::flush_clean_exit();
        std::string after;
        t.expect(read_file(path_mesh, &after), "mesh store still readable");
        t.expect(before == after,
                 "a loaded (not dirty) store never rewrites the file");

        cache.put(k_res, p_res);  // fresh compile -> dirty
        DiePlanPersistentCache::flush_clean_exit();
        std::string merged;
        t.expect(read_file(path_mesh, &merged), "merged store readable");
        HeaderLayout lay;
        t.expect(parse_layout(merged, &lay), "merged header parses");
        t.expect_uint(lay.count, 4, "merge grew the store to four entries");

        fresh_process(path_mesh, mesh);
        t.expect(DiePlanPersistentCache::instance().get(k_norm, mesh) !=
                     nullptr,
                 "old key still hits after the merge");
        t.expect(DiePlanPersistentCache::instance().get(k_res, mesh) !=
                     nullptr,
                 "new key hits after the merge");
    }

    // -----------------------------------------------------------------
    // Group 4: corruption matrix (fail-closed whole-file discard)
    {
        std::string good;
        t.expect(read_file(path_mesh, &good), "corruption base readable");
        t.expect(good.size() > 64, "corruption base is non-trivial");
        HeaderLayout lay;
        t.expect(parse_layout(good, &lay), "corruption base header parses");

        // Positive control: rewriting the file with only the checksum
        // recomputed must still load (the surgery helper is sound).
        {
            std::string ctrl = good;
            fix_checksum(&ctrl);
            const std::string path = unique_path("ctrl");
            t.expect(write_file(path, ctrl), "control file written");
            fresh_process(path, mesh);
            t.expect(DiePlanPersistentCache::instance().get(k_norm, mesh) !=
                         nullptr,
                     "positive control: recomputed checksum still loads");
            ::remove(path.c_str());
        }

        struct Case {
            const char* name;
            std::string bytes;
        };
        std::vector<Case> cases;
        {
            std::string c = good;
            c.resize(c.size() - 10);
            cases.push_back({"truncated tail", c});
        }
        {
            std::string c = good;
            c.resize(c.size() / 2);
            cases.push_back({"truncated mid-entry", c});
        }
        {
            std::string c = good;
            c.push_back('X');
            cases.push_back({"trailing garbage", c});
        }
        {
            std::string c = good;
            c[0] = static_cast<char>(c[0] ^ 0x01);
            fix_checksum(&c);
            cases.push_back({"wrong magic (checksum fixed)", c});
        }
        {
            std::string c = good;
            c[lay.fp_off] = static_cast<char>(c[lay.fp_off] ^ 0x01);
            fix_checksum(&c);
            cases.push_back({"wrong fingerprint (checksum fixed)", c});
        }
        {
            std::string c = good;
            c[lay.hw_off] = static_cast<char>(c[lay.hw_off] ^ 0x01);
            fix_checksum(&c);
            cases.push_back({"wrong hardware key (checksum fixed)", c});
        }
        {
            std::string c = good;
            c[c.size() - 1] =
                static_cast<char>(c[c.size() - 1] ^ 0x01);
            cases.push_back({"wrong checksum", c});
        }
        {
            // First entry, model_dims.hidden (feeds plan_cache_key): the
            // per-entry content-addressing self-check must fire.
            std::string c = good;
            const std::size_t hidden_off =
                lay.entries_off + 1 + 16 + 8 + 3 + 7;
            t.expect(hidden_off < c.size(), "entry mutation offset in range");
            c[hidden_off] = static_cast<char>(c[hidden_off] ^ 0x02);
            fix_checksum(&c);
            cases.push_back({"entry key self-check (checksum fixed)", c});
        }

        for (const Case& cs : cases) {
            const std::string path = unique_path("corrupt");
            t.expect(write_file(path, cs.bytes),
                     std::string("wrote corruption case ") + cs.name);
            fresh_process(path, mesh);
            DiePlanPersistentCache& cache =
                DiePlanPersistentCache::instance();
            t.expect(cache.get(k_norm, mesh) == nullptr &&
                         cache.get(k_mlp, mesh) == nullptr,
                     std::string("corrupt store discarded whole-file: ") +
                         cs.name);
            DiePlanPersistentCache::flush_clean_exit();
            std::string still;
            t.expect(read_file(path, &still) && still == cs.bytes,
                     std::string("rejected load leaves the file untouched: "
                                 "") +
                         cs.name);
            // Self-healing: a fresh compile in the same (now empty) store
            // persists normally over the rejected file.
            cache.put(k_norm, p_norm);
            DiePlanPersistentCache::flush_clean_exit();
            std::string healed;
            t.expect(read_file(path, &healed) && healed != cs.bytes,
                     std::string("fresh compile heals the store: ") +
                         cs.name);
            fresh_process(path, mesh);
            t.expect(DiePlanPersistentCache::instance().get(k_norm, mesh) !=
                         nullptr,
                     std::string("healed store hits again: ") + cs.name);
            ::remove(path.c_str());
        }
    }

    // -----------------------------------------------------------------
    // Group 5: switch discipline
    {
        const std::string path = unique_path("disabled");
        ::remove(path.c_str());
        DiePlanPersistentCache::reset_for_test();
        ::unsetenv("DIE_PLAN_CACHE_PATH");
        {
            DiePlanPersistentCache& cache =
                DiePlanPersistentCache::instance();
            t.expect(!cache.enabled(),
                     "unset env keeps the store disabled");
            t.expect(cache.get(k_norm, mesh) == nullptr,
                     "unset env lookup is null");
            cache.put(k_norm, p_norm);
            DiePlanPersistentCache::flush_clean_exit();
        }
        {
            ::setenv("DIE_PLAN_CACHE_PATH", "", 1);
            DiePlanPersistentCache::reset_for_test();
            DiePlanPersistentCache& cache =
                DiePlanPersistentCache::instance();
            t.expect(!cache.enabled(),
                     "empty env value keeps the store disabled");
            t.expect(cache.get(k_norm, mesh) == nullptr,
                     "empty env lookup is null");
            DiePlanPersistentCache::flush_clean_exit();
        }
        std::ifstream probe(path);
        t.expect(!probe, "disabled store never creates a file");
        ::unsetenv("DIE_PLAN_CACHE_PATH");
    }
    {
        // Set but absent file: empty store semantics, created on save.
        const std::string path = unique_path("absent");
        ::remove(path.c_str());
        fresh_process(path, mesh);
        DiePlanPersistentCache& cache = DiePlanPersistentCache::instance();
        t.expect(cache.enabled(), "absent-file store is still enabled");
        t.expect(cache.get(k_norm, mesh) == nullptr,
                 "absent file yields no hits");
        cache.put(k_norm, p_norm);
        DiePlanPersistentCache::flush_clean_exit();
        std::string buf;
        t.expect(read_file(path, &buf),
                 "put+flush creates the absent store file");
        fresh_process(path, mesh);
        t.expect(DiePlanPersistentCache::instance().get(k_norm, mesh) !=
                     nullptr,
                 "created store hits on the next process");
        ::remove(path.c_str());
    }

    // Cleanup + leave the process env pristine for later test phases.
    DiePlanPersistentCache::reset_for_test();
    ::unsetenv("DIE_PLAN_CACHE_PATH");
    ::remove(path_mesh.c_str());
    ::remove(path_tx8.c_str());

    if (t.ok) {
        std::printf("[test_die_plan_persistent_cache] ALL PASS\n");
        return 0;
    }
    return 1;
}
