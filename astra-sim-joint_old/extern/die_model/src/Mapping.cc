/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
Mapping.cc -- target-die tiling, capacity, residency and NoC mapping
(design doc §3.2 NoC / §5.1 candidate sets / §5.2 GEMM / §5.3 vector ops /
§5.4 fused FFN / §5.5 on-tile attention).

Produces one detail::Candidate per feasible mapping (repeat-compressed stage
DAG + exact totals); src/CostModel.cc evaluates every candidate with the SAME
isolated-run stage model (detail::evaluate_stages) and keeps the shortest
(design doc §5.1).  Semantic rulings frozen in the stage A2 report:

  R-W   weight HBM traffic = per-layer weight bytes x max(weight_passes,
        tiling-implied wave reloads); the weight-resident loop reads once per
        pass.  Macro-group launch instances = layer_count x weight_passes.
  R-N   wave-uniform NoC approximation: a stage's per-instance edge bytes
        scale its unit edge map by the average instance bytes.
  R-L   buffer-release lag = contexts when double-buffered, else 1.
  R-KV  K/V reload count = per span ceil(q_tokens/q_tile) sweeps x
        ceil(q_heads_local*k_parallel/group_cores) head passes.
*/

#include "DieModel.hh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace DieModel {
namespace detail {

// Implemented in CostModel.cc; shared with the FFN fallback sub-mapping
// selection (same stage model, design doc §5.4 "无可行融合映射则退回").
ScheduleResult evaluate_stages(const std::vector<DieStage>& stages,
                               const Hardware& hw);

namespace {

using u64 = std::uint64_t;

// Aggregated directed-edge load of a unit transfer (per one byte): stages
// only ever scale ONE unit map uniformly, so the per-edge map is folded
// into (max_edge, total) at construction time (ruling R-N).
struct EdgeAgg {
  double max_edge = 0.0;
  double total = 0.0;
};

u64 ceil_u64(u64 a, u64 b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// §5.1 candidate sets

const std::vector<u64> kTokenTiles = {1, 8, 16, 32, 64, 128};
const std::vector<u64> kOutputTiles = {32, 64, 128, 256, 512};
const std::vector<u64> kReductionTiles = {32, 64, 128, 256, 512};

std::vector<u64> tiles_for(u64 dim, const std::vector<u64>& base) {
  std::set<u64> s;
  for (u64 v : base) {
    if (v <= dim && v >= 1) {
      s.insert(v);
    }
  }
  // "补入小于最小候选的尺寸并去重" (§5.1): only dimensions smaller than
  // every candidate are supplemented; larger dimensions keep the candidate
  // sizes and run their tail tile at the true size (§5.1).
  if (s.empty()) {
    s.insert(dim < 1 ? 1 : dim);
  }
  return std::vector<u64>(s.begin(), s.end());
}

// Core group side choices {1,2,4,8,n} clamped to the mesh edge and to the
// number of work units in that direction (§5.1).
std::vector<u64> group_sides(u64 max_units, u64 mesh_n) {
  std::set<u64> s;
  s.insert(1);
  const u64 cap = std::min(mesh_n, max_units < 1 ? 1 : max_units);
  for (u64 v : {2u, 4u, 8u}) {
    if (v <= cap) {
      s.insert(v);
    }
  }
  s.insert(cap);
  return std::vector<u64>(s.begin(), s.end());
}

// ---------------------------------------------------------------------------
// §6.2 memory request with integer-tail striping across HBM stacks

DieMemoryRequest make_mem(double bytes, bool matrix, const Hardware& hw) {
  DieMemoryRequest r;
  r.actual_bytes = bytes;
  r.matrix_class = matrix;
  r.eta = matrix ? hw.eta_matrix_mem : hw.eta_vector_mem;
  const double n = static_cast<double>(hw.hbm_count);
  if (bytes <= 0.0 || n <= 0.0) {
    return r;
  }
  const double base = std::floor(bytes / n);
  const double tail = bytes - base * n;  // first stacks take the tail bytes
  double sum_u = 0.0;
  double max_u = 0.0;
  for (double j = 0.0; j < n; j += 1.0) {
    const double dj = base + (j < tail ? 1.0 : 0.0);
    const double uj = dj / r.eta;
    sum_u += uj;
    max_u = std::max(max_u, uj);
  }
  r.service_units = std::max(sum_u, n * max_u);  // §6.2 granularity rule
  return r;
}

// ---------------------------------------------------------------------------
// §3.2 NoC: fixed XY routing; striped injection at each stack's edge port
// nearest the target core group center (ties -> smaller column); broadcasts
// are directed-edge unions.  Edge key = ((y*mesh_n + x) << 2) | dir with
// dir 0=+x, 1=-x, 2=+y, 3=-y (all keys are "from core (x,y)").

class NocRouter {
 public:
  explicit NocRouter(const Hardware& hw)
      : hw_(hw),
        n_(hw.mesh_n),
        edge_count_(4 * hw.mesh_n * hw.mesh_n),
        scratch_(edge_count_, 0.0),
        stamps_(edge_count_, UINT64_MAX) {}

  // Fills the flat scratch edge array and folds it into an EdgeAgg.

  // ---- unit edge maps (per one byte transferred), cached per shape ----

  // Partitioned striped load: each stack delivers its 1/N share, split over
  // the group cores (1/G each), from its chosen port.
  std::shared_ptr<const EdgeAgg> load_unit(u64 rows, u64 cols) {
    return cached(rows, cols, 0, [this, rows, cols](EdgeAgg& agg) {
      const auto ports = stack_ports(rows, cols);
      const double share = 1.0 / (static_cast<double>(ports.size()) *
                                  static_cast<double>(rows * cols));
      scratch_.assign(edge_count_, 0.0);
      for (const PortXY& p : ports) {
        for (u64 i = 0; i < rows; ++i) {
          for (u64 j = 0; j < cols; ++j) {
            add_path(p.x, p.y, i, j, share);
          }
        }
      }
      fold(agg);
    });
  }

  // Broadcast striped load: the union of port->core paths; each directed
  // edge carries its stack's 1/N share once (§3.2).
  std::shared_ptr<const EdgeAgg> broadcast_unit(u64 rows, u64 cols) {
    return cached(rows, cols, 1, [this, rows, cols](EdgeAgg& agg) {
      const auto ports = stack_ports(rows, cols);
      const double share = 1.0 / static_cast<double>(ports.size());
      scratch_.assign(edge_count_, 0.0);
      for (const PortXY& p : ports) {
        const std::uint64_t before = stamp_++;
        for (u64 i = 0; i < rows; ++i) {
          for (u64 j = 0; j < cols; ++j) {
            add_path_once(p.x, p.y, i, j, share, before);
          }
        }
      }
      fold(agg);
    });
  }

  // Partitioned striped store (core -> its stack ports).
  std::shared_ptr<const EdgeAgg> store_unit(u64 rows, u64 cols) {
    return cached(rows, cols, 2, [this, rows, cols](EdgeAgg& agg) {
      const auto ports = stack_ports(rows, cols);
      const double share = 1.0 / (static_cast<double>(ports.size()) *
                                  static_cast<double>(rows * cols));
      scratch_.assign(edge_count_, 0.0);
      for (const PortXY& p : ports) {
        for (u64 i = 0; i < rows; ++i) {
          for (u64 j = 0; j < cols; ++j) {
            add_path(i, j, p.x, p.y, share);
          }
        }
      }
      fold(agg);
    });
  }

  // Fixed binary-tree merge over kp participating cores (linear ids 0..kp-1
  // of the group, row-major): at each level core i+half sends to core i
  // (§5.2/§5.5).
  std::shared_ptr<const EdgeAgg> merge_unit(u64 rows, u64 cols, u64 kp) {
    return cached(rows, cols, 16 + kp, [this, rows, cols, kp](EdgeAgg& agg) {
      scratch_.assign(edge_count_, 0.0);
      u64 k = kp;
      while (k > 1) {
        const u64 half = k / 2;
        for (u64 i = 0; i < half; ++i) {
          const u64 src = half + i;
          add_path(src / cols, src % cols, i / cols, i % cols, 1.0);
        }
        k = half;
      }
      fold(agg);
    });
  }

 private:
  template <typename F>
  std::shared_ptr<const EdgeAgg> cached(u64 rows, u64 cols, u64 kind, F fill) {
    const u64 key = ((rows * 128 + cols) << 6) | kind;
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      return it->second;
    }
    auto agg = std::make_shared<EdgeAgg>();
    fill(*agg);
    cache_[key] = agg;
    return agg;
  }

  void fold(EdgeAgg& agg) {
    double mx = 0.0, tot = 0.0;
    for (double v : scratch_) {
      if (v > 0.0) {
        mx = std::max(mx, v);
        tot += v;
      }
    }
    agg.max_edge = mx;
    agg.total = tot;
  }

  struct PortXY {
    u64 x, y;
  };

  // Per-stack injection port columns for the group at top-left rows x cols:
  // the port of the stack's contiguous edge segment nearest the group center
  // (Manhattan distance from the off-die port, ties -> smaller column, §3.2).
  // Stack s < hbm_per_side sits on the top edge, the rest on the bottom.
  std::vector<PortXY> stack_ports(u64 rows, u64 cols) const {
    const u64 ps = hw_.hbm_per_side;
    const long long cx = static_cast<long long>((cols - 1) / 2);
    const long long cy = static_cast<long long>((rows - 1) / 2);
    std::vector<PortXY> out;
    for (u64 s = 0; s < hw_.hbm_count; ++s) {
      const bool bottom = s >= ps;
      const u64 seg = bottom ? s - ps : s;
      const u64 c0 = seg * n_ / ps;
      const u64 c1 = (seg + 1) * n_ / ps;
      u64 best = c0;
      long long best_d = -1;
      for (u64 c = c0; c < c1; ++c) {
        long long d = std::llabs(static_cast<long long>(c) - cx);
        d += bottom ? static_cast<long long>(n_ - 1) - cy : cy;
        if (best_d < 0 || d < best_d) {
          best_d = d;
          best = c;
        }
      }
      out.push_back(PortXY{best, bottom ? n_ - 1 : 0});
    }
    return out;
  }

  // Accumulating variant (partitioned transfers: every byte on the path).
  void add_path(u64 sx, u64 sy, u64 tx, u64 ty, double bytes) {
    u64 x = sx, y = sy;
    while (x != tx) {  // X first (fixed XY routing, §3.2)
      const bool east = tx > x;
      scratch_[edge_key(x, y, east ? 0 : 1)] += bytes;
      x = east ? x + 1 : x - 1;
    }
    while (y != ty) {
      const bool south = ty > y;
      scratch_[edge_key(x, y, south ? 2 : 3)] += bytes;
      y = south ? y + 1 : y - 1;
    }
  }

  // Union variant for broadcasts: each directed edge counts the stack share
  // once per stack even when several destination paths share it (§3.2).  A
  // per-edge stamp ensures one count per stack pass.
  void add_path_once(u64 sx, u64 sy, u64 tx, u64 ty, double bytes,
                     std::uint64_t stamp) {
    u64 x = sx, y = sy;
    while (x != tx) {
      const bool east = tx > x;
      stamp_edge(edge_key(x, y, east ? 0 : 1), bytes, stamp);
      x = east ? x + 1 : x - 1;
    }
    while (y != ty) {
      const bool south = ty > y;
      stamp_edge(edge_key(x, y, south ? 2 : 3), bytes, stamp);
      y = south ? y + 1 : y - 1;
    }
  }

  void stamp_edge(u64 key, double bytes, std::uint64_t stamp) {
    if (stamps_[key] != stamp) {
      stamps_[key] = stamp;
      scratch_[key] += bytes;
    }
  }

  u64 edge_key(u64 x, u64 y, u64 dir) const {
    return ((y * n_ + x) << 2) | dir;
  }

  const Hardware& hw_;
  u64 n_;
  const std::size_t edge_count_;
  std::vector<double> scratch_;
  std::vector<std::uint64_t> stamps_;  // last broadcast-stamp per edge
  std::uint64_t stamp_ = 0;
  std::map<u64, std::shared_ptr<const EdgeAgg>> cache_;
};

// ---------------------------------------------------------------------------
// Assembly context and helpers

struct Ctx {
  const Hardware& hw;
  NocRouter& noc;
  double limit;  // per-core active-buffer budget (0.90 usable, §5.1)
};

void add_launch(Candidate& c, const Ctx& ctx, u64 macro, u64 layers,
                u64 passes) {
  DieStage s;
  s.label = "launch";
  s.resource = StageResource::TIMER;
  s.macro_group = macro;
  s.wave_repeat = layers * passes;
  s.timer_seconds = ctx.hw.launch_us_per_macro_group * 1e-6;
  c.summary.launch_instances = layers * passes;
  c.stages.push_back(std::move(s));
}


// Registers one MEM_NOC stage and folds in its NoC edge accounting (§3.2).
size_t add_mem_stage(Candidate& c, const Ctx& ctx, const char* label, u64 macro,
                     u64 repeat, double bytes_per_instance, bool matrix,
                     std::shared_ptr<const EdgeAgg> unit,
                     std::vector<DieStageDep> deps) {
  DieStage s;
  s.label = label;
  s.resource = StageResource::MEM_NOC;
  s.macro_group = macro;
  s.wave_repeat = repeat;
  s.mem = make_mem(bytes_per_instance, matrix, ctx.hw);
  s.deps = std::move(deps);
  if (repeat > 0 && bytes_per_instance > 0.0 && unit) {
    s.noc_edge_bytes = unit->max_edge * bytes_per_instance;
    c.noc_total_link_bytes +=
        unit->total * bytes_per_instance * static_cast<double>(repeat);
    c.noc_max_edge_bytes = std::max(c.noc_max_edge_bytes, s.noc_edge_bytes);
  }
  if (matrix) {
    c.matrix_bytes += bytes_per_instance * static_cast<double>(repeat);
    c.matrix_service_units += s.mem.service_units * static_cast<double>(repeat);
  } else {
    c.vector_bytes += bytes_per_instance * static_cast<double>(repeat);
    c.vector_service_units += s.mem.service_units * static_cast<double>(repeat);
  }
  c.stages.push_back(std::move(s));
  return c.stages.size() - 1;
}

size_t add_compute_stage(Candidate& c, const Ctx& ctx, const char* label,
                         StageResource res, u64 macro, u64 repeat, double work,
                         std::vector<DieStageDep> deps,
                         std::shared_ptr<const EdgeAgg> noc_unit = nullptr,
                         double noc_scale = 0.0) {
  DieStage s;
  s.label = label;
  s.resource = res;
  s.macro_group = macro;
  s.wave_repeat = repeat;
  if (res == StageResource::PE) {
    s.pe_core_flop = work;
  } else if (res == StageResource::VEC) {
    s.vec_core_eqop = work;
  }
  s.deps = std::move(deps);
  if (noc_unit && repeat > 0 && noc_scale > 0.0) {
    s.noc_edge_bytes = noc_unit->max_edge * noc_scale;
    c.noc_total_link_bytes +=
        noc_unit->total * noc_scale * static_cast<double>(repeat);
    c.noc_max_edge_bytes = std::max(c.noc_max_edge_bytes, s.noc_edge_bytes);
  }
  c.stages.push_back(std::move(s));
  return c.stages.size() - 1;
}

// ---------------------------------------------------------------------------
// §5.2 GEMM mapping

struct GemmJob {
  u64 M = 0, K = 0, N = 0;
  double elem = 2.0;
  u64 layers = 1, passes = 1;
  u64 macro = 0;
  // Optional fused vector tail (the RoPE tail of qkv_rope): per-layer eq-ops
  // and aux bytes (vector class) executed on the same core group.
  bool has_tail = false;
  double tail_eqops = 0.0;
  double tail_bytes = 0.0;
};

double sram_gemm(double e, u64 m, u64 k, u64 n, bool dbl) {
  // S = 4mk+4kn+4mn double / 2mk+2kn+4mn single at BF16 (§5.2), generalized
  // by element size; the FP32 accumulator stays 4mn.
  const double mk = static_cast<double>(m * k);
  const double kn = static_cast<double>(k * n);
  const double mn = static_cast<double>(m * n);
  return dbl ? (2.0 * e * mk + 2.0 * e * kn + 4.0 * mn)
             : (e * mk + e * kn + 4.0 * mn);
}

u64 largest_k_tile(const Ctx& ctx, double e, u64 m, u64 n, u64 K, u64 ctx_count,
                   bool dbl) {
  const std::vector<u64> ks = tiles_for(K, kReductionTiles);
  for (auto it = ks.rbegin(); it != ks.rend(); ++it) {
    if (sram_gemm(e, m, *it, n, dbl) * ctx_count <= ctx.limit) {
      return *it;
    }
  }
  return 0;
}

void build_output_resident(const Ctx& ctx, const GemmJob& j, u64 m, u64 n,
                           u64 rows, u64 cols, u64 kp, u64 ctx_count, bool dbl,
                           u64 k_tile, Candidate& c) {
  const double e = j.elem;
  const double M = static_cast<double>(j.M);
  const double K = static_cast<double>(j.K);
  const double N = static_cast<double>(j.N);
  const double L = static_cast<double>(j.layers);
  const double G = static_cast<double>(rows * cols);
  const u64 rb = rows / kp;  // row blocks per wave (K-slices down a column)
  const u64 T_m = ceil_u64(j.M, m);
  const u64 T_n = ceil_u64(j.N, n);
  const u64 row_sweeps = ceil_u64(T_m, rb);
  const u64 waves = row_sweeps * ceil_u64(T_n, cols);

  add_launch(c, ctx, j.macro, j.layers, j.passes);
  const size_t launch = 0;

  // A row blocks are read once per wave and broadcast over the columns; W
  // column blocks once per wave, broadcast over the row blocks.  Weight HBM
  // traffic follows ruling R-W.
  const double a_wave = static_cast<double>(rb) * e * static_cast<double>(m) * K;
  const double w_wave = static_cast<double>(cols) * e * static_cast<double>(n) * K;
  const double a_total = a_wave * static_cast<double>(waves) * L;
  const double w_charged =
      e * K * N * static_cast<double>(std::max(j.passes, row_sweeps)) * L;
  const u64 load_repeat = waves * j.layers;
  const size_t load = add_mem_stage(
      c, ctx, "load", j.macro, load_repeat,
      (a_total + w_charged) / static_cast<double>(load_repeat), true,
      ctx.noc.broadcast_unit(rows, cols), {{launch, DieStageDep::kAllDep}});

  const u64 kslice = ceil_u64(j.K, kp);
  const size_t pe = add_compute_stage(
      c, ctx, "pe", StageResource::PE, j.macro, load_repeat,
      2.0 * static_cast<double>(m * n) * static_cast<double>(kslice),
      {{load, 0}});

  size_t last_compute = pe;
  if (kp > 1) {
    // K-split partial-C reduction: 4mn-level messages + (kp-1)*mn vector
    // adds per tile (§5.2).
    const auto unit = ctx.noc.merge_unit(rows, cols, kp);
    const double msg = 4.0 * static_cast<double>(m * n) *
                       static_cast<double>(kp - 1);
    const size_t red =
        add_compute_stage(c, ctx, "reduce_noc", StageResource::NOC, j.macro,
                          load_repeat, 0.0, {{pe, 0}}, unit, msg);
    const size_t rv = add_compute_stage(
        c, ctx, "reduce_vec", StageResource::VEC, j.macro, load_repeat,
        static_cast<double>(m * n) * static_cast<double>(kp - 1) / G, {{pe, 0}});
    c.eqops += static_cast<double>(m * n) * static_cast<double>(kp - 1) *
               static_cast<double>(load_repeat);
    last_compute = rv;
    (void)red;
  }

  size_t tail_vec = static_cast<size_t>(-1);
  if (j.has_tail) {
    const size_t tail_load =
        add_mem_stage(c, ctx, "tail_load", j.macro, j.layers, j.tail_bytes,
                      false, ctx.noc.load_unit(rows, cols),
                      {{launch, DieStageDep::kAllDep}});
    tail_vec = add_compute_stage(
        c, ctx, "tail_vec", StageResource::VEC, j.macro, j.layers,
        j.tail_eqops / G,
        {{last_compute, DieStageDep::kAllDep}, {tail_load, DieStageDep::kAllDep}});
    c.eqops += j.tail_eqops * L;
  }

  const double out_total = e * M * N * L;
  std::vector<DieStageDep> wdeps;
  if (tail_vec != static_cast<size_t>(-1)) {
    wdeps.push_back({tail_vec, DieStageDep::kAllDep});
  } else {
    wdeps.push_back({last_compute, 0});
  }
  add_mem_stage(c, ctx, "write", j.macro, load_repeat,
                out_total / static_cast<double>(load_repeat), true,
                ctx.noc.store_unit(rows, cols), std::move(wdeps));

  // Buffer release: the next wave's load starts once the compute of `lag`
  // waves before it released its tile buffers (ruling R-L).
  const u64 lag = dbl ? ctx_count : 1;
  c.stages[load].deps.push_back(
      {static_cast<std::uint32_t>(pe), static_cast<std::uint32_t>(lag)});

  c.flops = 2.0 * M * K * N * L;
  c.summary.m_tile = m;
  c.summary.n_tile = n;
  c.summary.k_tile = k_tile;
  c.summary.group_rows = rows;
  c.summary.group_cols = cols;
  c.summary.contexts = ctx_count;
  c.summary.k_parallel = kp;
  c.summary.weight_resident = false;
  c.summary.double_buffered = dbl;
  c.summary.layer_count = j.layers;
  c.summary.weight_passes = j.passes;
  c.summary.waves = waves;
  c.summary.weight_hbm_bytes = w_charged;
  c.summary.sram_peak_per_core =
      sram_gemm(e, m, k_tile, n, dbl) * static_cast<double>(ctx_count);
  c.summary.loop_kind = "gemm_output_resident";
}

void build_weight_resident(const Ctx& ctx, const GemmJob& j, u64 m, u64 n,
                           u64 k, u64 cols, u64 kp, u64 ctx_count, bool dbl,
                           Candidate& c) {
  const double e = j.elem;
  const double M = static_cast<double>(j.M);
  const double K = static_cast<double>(j.K);
  const double N = static_cast<double>(j.N);
  const double L = static_cast<double>(j.layers);
  const double G = static_cast<double>(kp * cols);
  const u64 rows = kp;  // K-parallel rows; one n-tile column each (§5.2)
  const u64 T_k = ceil_u64(j.K, k);
  const u64 T_n = ceil_u64(j.N, n);
  const u64 m_waves = ceil_u64(j.M, m);
  const u64 sweeps = ceil_u64(T_n, cols);
  const u64 waves = T_k * m_waves * sweeps;

  add_launch(c, ctx, j.macro, j.layers, j.passes);
  const size_t launch = 0;

  const double a_total = e * M * K * static_cast<double>(sweeps) * L;
  const double w_charged = e * K * N * static_cast<double>(j.passes) * L;
  const u64 load_repeat = waves * j.layers;
  const size_t load = add_mem_stage(
      c, ctx, "load", j.macro, load_repeat,
      (a_total + w_charged) / static_cast<double>(load_repeat), true,
      ctx.noc.broadcast_unit(rows, cols), {{launch, DieStageDep::kAllDep}});

  const u64 kslice = ceil_u64(k, kp);
  const size_t pe = add_compute_stage(
      c, ctx, "pe", StageResource::PE, j.macro, load_repeat,
      2.0 * static_cast<double>(m * n) * static_cast<double>(kslice),
      {{load, 0}});

  size_t last_compute = pe;
  if (kp > 1) {
    const auto unit = ctx.noc.merge_unit(rows, cols, kp);
    const double msg = 4.0 * static_cast<double>(m * n) *
                       static_cast<double>(kp - 1);
    add_compute_stage(c, ctx, "reduce_noc", StageResource::NOC, j.macro,
                      load_repeat, 0.0, {{pe, 0}}, unit, msg);
    const size_t rv = add_compute_stage(
        c, ctx, "reduce_vec", StageResource::VEC, j.macro, load_repeat,
        static_cast<double>(m * n) * static_cast<double>(kp - 1) / G, {{pe, 0}});
    c.eqops += static_cast<double>(m * n) * static_cast<double>(kp - 1) *
               static_cast<double>(load_repeat);
    last_compute = rv;
  }

  // Output: without a resident [M, n] FP32 accumulator slice, every K-block
  // boundary reads+writes 4-byte partial C (first K no read, last K writes
  // BF16, §5.2).
  const bool resident =
      4.0 * M * static_cast<double>(n) +
              sram_gemm(e, m, k, n, dbl) * static_cast<double>(ctx_count) <=
          ctx.limit;
  if (!resident) {
    add_mem_stage(c, ctx, "partial_spill", j.macro, T_n * (T_k - 1) * j.layers,
                  8.0 * M * static_cast<double>(n), true,
                  ctx.noc.store_unit(rows, cols),
                  {{last_compute, DieStageDep::kAllDep}});
  }
  const double out_total = e * M * N * L;
  add_mem_stage(c, ctx, "write", j.macro, m_waves * T_n * j.layers,
                out_total / static_cast<double>(m_waves * T_n * j.layers), true,
                ctx.noc.store_unit(rows, cols),
                {{last_compute, DieStageDep::kAllDep}});

  c.flops = 2.0 * M * K * N * L;
  c.summary.m_tile = m;
  c.summary.n_tile = n;
  c.summary.k_tile = k;
  c.summary.group_rows = rows;
  c.summary.group_cols = cols;
  c.summary.contexts = ctx_count;
  c.summary.k_parallel = kp;
  c.summary.weight_resident = true;
  c.summary.double_buffered = dbl;
  c.summary.layer_count = j.layers;
  c.summary.weight_passes = j.passes;
  c.summary.waves = waves;
  c.summary.weight_hbm_bytes = w_charged;
  c.summary.sram_peak_per_core =
      std::max(sram_gemm(e, m, k, n, dbl) * static_cast<double>(ctx_count),
               resident ? 4.0 * M * static_cast<double>(n) : 0.0);
  c.summary.loop_kind = "gemm_weight_resident";
}

std::vector<Candidate> map_gemm(const Ctx& ctx, const GemmJob& j) {
  std::vector<Candidate> out;
  if (j.M == 0 || j.K == 0 || j.N == 0) {
    return out;
  }
  const double e = j.elem;
  for (u64 m : tiles_for(j.M, kTokenTiles)) {
    const u64 T_m = ceil_u64(j.M, m);
    for (u64 n : tiles_for(j.N, kOutputTiles)) {
      const u64 T_n = ceil_u64(j.N, n);
      for (u64 kp : {1u, 2u, 4u, 8u}) {
        if (kp > 1 && kp > j.K) {
          continue;
        }
        for (u64 rb : group_sides(T_m, ctx.hw.mesh_n)) {
          const u64 rows = kp * rb;
          if (rows > ctx.hw.mesh_n) {
            continue;
          }
          for (u64 cols : group_sides(T_n, ctx.hw.mesh_n)) {
            for (u64 ctx_count : {1u, 2u}) {
              u64 k_tile = largest_k_tile(ctx, e, m, n, j.K, ctx_count, true);
              bool dbl = true;
              if (k_tile == 0) {
                k_tile = largest_k_tile(ctx, e, m, n, j.K, ctx_count, false);
                dbl = false;
              }
              if (k_tile == 0) {
                continue;  // smaller (m,n) cover this case (§5.1)
              }
              Candidate c;
              build_output_resident(ctx, j, m, n, rows, cols, kp, ctx_count,
                                    dbl, k_tile, c);
              out.push_back(std::move(c));
            }
          }
        }
        if (kp <= ctx.hw.mesh_n) {
          for (u64 cols : group_sides(T_n, ctx.hw.mesh_n)) {
            for (u64 ctx_count : {1u, 2u}) {
              for (bool dbl : {true, false}) {
                const std::vector<u64> ks = tiles_for(j.K, kReductionTiles);
                for (auto kit = ks.rbegin(); kit != ks.rend(); ++kit) {
                  if (sram_gemm(e, m, *kit, n, dbl) * ctx_count > ctx.limit) {
                    continue;
                  }
                  Candidate c;
                  build_weight_resident(ctx, j, m, n, *kit, cols, kp,
                                        ctx_count, dbl, c);
                  out.push_back(std::move(c));
                  break;  // largest feasible k only
                }
              }
            }
          }
        }
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// §5.3 vector-op mapping

struct VectorJob {
  u64 rows_count = 0;   // logical rows (tokens or score rows)
  u64 row_len = 0;      // elements per row (H, or a span's kv length)
  double bytes_in_per_layer = 0.0;
  double bytes_out_per_layer = 0.0;
  double weight_row_bytes = 0.0;  // gamma/beta row (0 if none)
  double eqops_per_layer = 0.0;
  u64 layers = 1, passes = 1;
  u64 macro = 0;
};

void build_vector(const Ctx& ctx, const VectorJob& j, u64 m, u64 rows, u64 cols,
                  u64 ctx_count, bool resident, u64 chunk, Candidate& c) {
  const double L = static_cast<double>(j.layers);
  const double G = static_cast<double>(rows * cols);
  const u64 rows_per_core = ceil_u64(j.rows_count, rows * cols);
  const u64 waves = ceil_u64(rows_per_core, m);
  const double per_row_in =
      j.bytes_in_per_layer / static_cast<double>(j.rows_count);
  const double per_row_out =
      j.bytes_out_per_layer / static_cast<double>(j.rows_count);

  add_launch(c, ctx, j.macro, j.layers, j.passes);
  const size_t launch = 0;

  const double in_total =
      j.bytes_in_per_layer * (resident ? 1.0 : 2.0) * L;  // §5.3 two-pass
  const double w_total = j.weight_row_bytes * static_cast<double>(j.passes) * L;
  const u64 repeat = waves * j.layers;
  const size_t load = add_mem_stage(
      c, ctx, "load", j.macro, repeat,
      (in_total + w_total) / static_cast<double>(repeat), false,
      ctx.noc.load_unit(rows, cols), {{launch, DieStageDep::kAllDep}});

  const size_t vec = add_compute_stage(
      c, ctx, "vec", StageResource::VEC, j.macro, repeat,
      j.eqops_per_layer * L / (G * static_cast<double>(waves)), {{load, 0}});
  c.eqops += j.eqops_per_layer * L;

  const double out_total = j.bytes_out_per_layer * L;
  add_mem_stage(c, ctx, "write", j.macro, repeat,
                out_total / static_cast<double>(repeat), false,
                ctx.noc.store_unit(rows, cols), {{vec, 0}});
  // Buffer release of the pipelined row buffers (ruling R-L; vector row
  // buffers are single-structure, so the lag is the context count).
  c.stages[load].deps.push_back(
      {static_cast<std::uint32_t>(vec), static_cast<std::uint32_t>(ctx_count)});

  c.summary.m_tile = m;
  c.summary.n_tile = j.row_len;
  c.summary.k_tile = resident ? j.row_len : chunk;
  c.summary.group_rows = rows;
  c.summary.group_cols = cols;
  c.summary.contexts = ctx_count;
  c.summary.layer_count = j.layers;
  c.summary.weight_passes = j.passes;
  c.summary.waves = waves;
  c.summary.streaming_two_pass = !resident;
  c.summary.weight_hbm_bytes = w_total;
  const double chunk_fraction =
      resident ? 1.0
               : static_cast<double>(chunk) / static_cast<double>(j.row_len);
  c.summary.sram_peak_per_core =
      (per_row_in + per_row_out) * chunk_fraction *
          static_cast<double>(ctx_count) +
      j.weight_row_bytes;
  c.summary.loop_kind =
      resident ? "vector_row_resident" : "vector_two_pass_stream";
}

std::vector<Candidate> map_vector(const Ctx& ctx, const VectorJob& j) {
  std::vector<Candidate> out;
  if (j.rows_count == 0 || j.row_len == 0 || j.eqops_per_layer <= 0.0) {
    return out;
  }
  const double per_row_in =
      j.bytes_in_per_layer / static_cast<double>(j.rows_count);
  const double per_row_out =
      j.bytes_out_per_layer / static_cast<double>(j.rows_count);
  for (u64 m : tiles_for(j.rows_count, kTokenTiles)) {
    for (u64 rows : group_sides(j.rows_count, ctx.hw.mesh_n)) {
      const u64 row_units = ceil_u64(j.rows_count, rows);
      for (u64 cols : group_sides(row_units, ctx.hw.mesh_n)) {
        for (u64 ctx_count : {1u, 2u}) {
          // Row-resident preferred: input + output + weight row in SRAM
          // (§5.3 "Norm 优先整行 H 驻留"); per_row_* are bytes per row.
          const double s_res =
              (per_row_in + per_row_out) * static_cast<double>(ctx_count) +
              j.weight_row_bytes;
          if (s_res <= ctx.limit) {
            Candidate c;
            build_vector(ctx, j, m, rows, cols, ctx_count, true, j.row_len, c);
            out.push_back(std::move(c));
            continue;
          }
          const std::vector<u64> chunks = tiles_for(j.row_len, kReductionTiles);
          for (auto it = chunks.rbegin(); it != chunks.rend(); ++it) {
            const double s_stream =
                (per_row_in + per_row_out) *
                    (static_cast<double>(*it) /
                     static_cast<double>(j.row_len)) *
                    static_cast<double>(ctx_count) +
                j.weight_row_bytes;
            if (s_stream <= ctx.limit) {
              Candidate c;
              build_vector(ctx, j, m, rows, cols, ctx_count, false, *it, c);
              out.push_back(std::move(c));
              break;
            }
          }
        }
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// §5.5 on-tile attention engine (fused region, QK-only, AV-only)

struct AttnJob {
  const WorkloadQuantities* wl = nullptr;
  u64 layers = 1, passes = 1;
  u64 macro = 0;
  enum class Mode { Full, QkOnly, AvOnly } mode = Mode::Full;
};

double sram_attn(double e, u64 q, u64 k, u64 d) {
  // S_attn = 2qd + 8kd + 4qk + 8qd + 16q at BF16 (§5.5), element-size
  // generalized for Q/K/V; scores/output partials/states stay FP32.
  return e * static_cast<double>(q * d) + 4.0 * e * static_cast<double>(k * d) +
         4.0 * static_cast<double>(q * k) + 8.0 * static_cast<double>(q * d) +
         16.0 * static_cast<double>(q);
}

// Tile walk of one KV slice with absolute-position causal skipping (only
// tiles entirely in the future are skipped; cross-diagonal tiles count their
// full rectangle, §4.2 rule 7 / §5.5).
struct UnitWalk {
  double pairs = 0.0;       // q*k elements of executed tiles
  double tiles = 0.0;
  double merge_eqops = 0.0;  // in-core sequential state merges
};

// Attention work shared by every core-group layout of one (qt, kt, kp)
// mapping axis.  Keep the accumulation order in aggregate_attention_work()
// identical to the former per-layout loop in build_attention().
struct AttentionWorkAggregate {
  u64 unit_count = 0;
  double total_flops = 0.0;
  double total_vec = 0.0;
  double total_pairs = 0.0;
  double max_unit_flops = 0.0;
  double max_unit_vec = 0.0;
};

UnitWalk walk_slice(const DieAttentionSpanWork& span, u64 q_rows, u64 q_abs_hi,
                    u64 lo, u64 hi, u64 kt, u64 d) {
  UnitWalk w;
  for (u64 l = lo; l < hi; l += kt) {
    const u64 h = std::min(l + kt, hi);
    if (span.kv_start_pos + l >= q_abs_hi) {
      continue;  // entirely in the future
    }
    const double cols = static_cast<double>(h - l);
    w.pairs += static_cast<double>(q_rows) * cols;
    if (w.tiles > 0.0) {
      w.merge_eqops +=
          static_cast<double>(q_rows) * (3.0 * static_cast<double>(d) + 22.0);
    }
    w.tiles += 1.0;
  }
  return w;
}

AttentionWorkAggregate aggregate_attention_work(
    const WorkloadQuantities& wl, AttnJob::Mode mode, u64 qt, u64 kt, u64 kp,
    EnumerationDiagnostics* diagnostics) {
  AttentionWorkAggregate work;
  if (diagnostics != nullptr) {
    ++diagnostics->attention_aggregate_builds;
  }
  const double d = static_cast<double>(wl.head_dim);
  for (const DieAttentionSpanWork& span : wl.attention_spans) {
    for (u64 qb = 0; qb < span.q_tokens; qb += qt) {
      const u64 q_rows = std::min(qt, span.q_tokens - qb);
      const u64 q_abs_hi = span.q_start_pos + qb + q_rows;  // exclusive
      double pairs = 0.0, merges = 0.0;
      for (u64 p = 0; p < kp; ++p) {
        const u64 total_tiles = ceil_u64(span.kv_tokens, kt);
        const u64 base = total_tiles / kp;
        const u64 rem = total_tiles % kp;
        const u64 len = base + (p < rem ? 1 : 0);
        const u64 lo = (p * base + std::min(p, rem)) * kt;
        const u64 hi = std::min(span.kv_tokens, lo + len * kt);
        if (diagnostics != nullptr) {
          ++diagnostics->attention_unit_walks;
        }
        const UnitWalk w =
            walk_slice(span, q_rows, q_abs_hi, lo, hi, kt, wl.head_dim);
        pairs += w.pairs;
        merges += w.merge_eqops;
      }
      // Matrix 4qkd + vector base 13qk + merges + tree merges + final
      // normalization q*d (§5.5); QkOnly/AvOnly are half the matrix with
      // materialized FP32 scores (§2.2 rows).
      double flops = 4.0 * pairs * d;
      double vec = 13.0 * pairs + merges +
                   static_cast<double>(kp - 1) * static_cast<double>(q_rows) *
                       (3.0 * d + 22.0) +
                   static_cast<double>(q_rows * wl.head_dim);
      if (mode == AttnJob::Mode::QkOnly || mode == AttnJob::Mode::AvOnly) {
        flops *= 0.5;
        vec = 0.0;
      }
      work.unit_count += wl.q_heads_local;
      const double unit_flops = flops * static_cast<double>(wl.q_heads_local);
      const double unit_vec = vec * static_cast<double>(wl.q_heads_local);
      work.total_flops += unit_flops;
      work.total_vec += unit_vec;
      work.total_pairs += pairs * static_cast<double>(wl.q_heads_local);
      work.max_unit_flops = std::max(work.max_unit_flops, unit_flops);
      work.max_unit_vec = std::max(work.max_unit_vec, unit_vec);
    }
  }
  return work;
}

void build_attention(const Ctx& ctx, const AttnJob& j, u64 qt, u64 kt, u64 rows,
                     u64 cols, u64 kp, u64 ctx_count,
                     const AttentionWorkAggregate& work, Candidate& c) {
  const WorkloadQuantities& wl = *j.wl;
  const double e = static_cast<double>(wl.elem_bytes);
  const double d = static_cast<double>(wl.head_dim);
  const double L = static_cast<double>(j.layers);
  const u64 G = rows * cols;
  const u64 CU = G / kp;  // concurrent units
  if (CU < 1) {
    return;
  }

  // This aggregate depends on the q/k tile and K/V partition only; per-layout
  // quantities (CU/G, K/V reloads and NoC routing) remain below.
  if (work.unit_count == 0 || work.total_pairs <= 0.0) {
    return;  // nothing executes (all tiles in the future)
  }
  const double cu = static_cast<double>(CU);
  const double max_flops = work.total_flops / cu + work.max_unit_flops;
  const double max_vec = work.total_vec / cu + work.max_unit_vec;

  // K/V reloads: per span, every q-tile sweep reloads its KV; heads of one
  // GQA group share a sweep when the core group allows (ruling R-KV).
  const double passes_h = std::ceil(static_cast<double>(wl.q_heads_local * kp) /
                                    static_cast<double>(G));
  const double kv_tensors =
      j.mode == AttnJob::Mode::Full ? 2.0 : 1.0;  // K+V / K-only / V-only
  double kv_total = 0.0;
  for (const DieAttentionSpanWork& span : wl.attention_spans) {
    kv_total += kv_tensors * e * d * static_cast<double>(wl.kv_heads_local) *
                static_cast<double>(span.kv_tokens) *
                static_cast<double>(ceil_u64(span.q_tokens, qt)) * passes_h;
  }

  const u64 waves = ceil_u64(work.unit_count, CU);
  add_launch(c, ctx, j.macro, j.layers, j.passes);
  const size_t launch = 0;
  const u64 repeat = waves * j.layers;

  const double q_bytes = e * static_cast<double>(wl.q_heads_local * wl.head_dim) *
                         static_cast<double>(wl.m_tokens);
  const double score_in =
      j.mode == AttnJob::Mode::AvOnly ? 4.0 * work.total_pairs * L : 0.0;
  const size_t load_q = add_mem_stage(
      c, ctx, "load_q", j.macro, j.layers, q_bytes + score_in, true,
      ctx.noc.load_unit(rows, cols), {{launch, DieStageDep::kAllDep}});

  const size_t load_kv = add_mem_stage(
      c, ctx, "load_kv", j.macro, repeat, kv_total * L / static_cast<double>(repeat),
      true, ctx.noc.broadcast_unit(rows, cols),
      {{launch, DieStageDep::kAllDep}});
  (void)load_q;

  const size_t pe = add_compute_stage(
      c, ctx, "attn_pe", StageResource::PE, j.macro, repeat,
      max_flops / static_cast<double>(waves), {{load_kv, 0}});

  size_t last = pe;
  if (j.mode == AttnJob::Mode::Full) {
    last = add_compute_stage(c, ctx, "attn_vec", StageResource::VEC, j.macro,
                             repeat, max_vec / static_cast<double>(waves),
                             {{pe, 0}});
    if (kp > 1) {
      // Cross-partition merge messages: 4q(d+2) bytes per q row per merge on
      // a fixed binary tree; every level pays messages AND vector merges
      // (§5.5); the vector side is already inside attn_vec above.
      const auto unit = ctx.noc.merge_unit(rows, cols, kp);
      const double msg = 4.0 * static_cast<double>(qt) * (d + 2.0) *
                         static_cast<double>(kp - 1) *
                         static_cast<double>(CU);
      add_compute_stage(c, ctx, "merge_noc", StageResource::NOC, j.macro,
                        repeat, 0.0, {{pe, 0}}, unit,
                        msg / static_cast<double>(CU));
      last = c.stages.size() - 1;
    }
  }

  const double o_out = e * static_cast<double>(wl.q_heads_local * wl.head_dim) *
                       static_cast<double>(wl.m_tokens) * L;
  const double score_out =
      j.mode == AttnJob::Mode::QkOnly ? 4.0 * work.total_pairs * L : 0.0;
  add_mem_stage(c, ctx, "write", j.macro, repeat,
                (o_out + score_out) / static_cast<double>(repeat), true,
                ctx.noc.store_unit(rows, cols), {{last, 0}});

  c.flops = work.total_flops * L;
  c.eqops = work.total_vec * L;
  c.summary.m_tile = qt;
  c.summary.n_tile = kt;
  c.summary.group_rows = rows;
  c.summary.group_cols = cols;
  c.summary.contexts = ctx_count;
  c.summary.k_parallel = kp;
  c.summary.fused = j.mode == AttnJob::Mode::Full;
  c.summary.layer_count = j.layers;
  c.summary.weight_passes = j.passes;
  c.summary.waves = waves;
  c.summary.sram_peak_per_core =
      sram_attn(e, qt, kt, wl.head_dim) * static_cast<double>(ctx_count);
  c.summary.loop_kind = "attention_tile";
}

std::vector<Candidate> map_attention(
    const Ctx& ctx, const AttnJob& j,
    EnumerationDiagnostics* diagnostics) {
  std::vector<Candidate> out;
  const WorkloadQuantities& wl = *j.wl;
  if (wl.q_heads_local == 0 || wl.attention_spans.empty() ||
      wl.head_dim == 0) {
    return out;
  }
  u64 q_max = 0, kv_max = 0;
  for (const auto& s : wl.attention_spans) {
    q_max = std::max(q_max, s.q_tokens);
    kv_max = std::max(kv_max, s.kv_tokens);
  }
  for (u64 qt : tiles_for(q_max, kTokenTiles)) {
    for (u64 kt : tiles_for(kv_max, kReductionTiles)) {
      for (u64 kp : {1u, 2u, 4u, 8u}) {
        const u64 tiles_min = ceil_u64(kv_max, kt);
        if (kp > 1 && kp > tiles_min) {
          continue;  // "限制为本组容量与实际工作允许" (§5.1)
        }
        bool have_work = false;
        AttentionWorkAggregate work;
        u64 unit_est = 0;
        for (const auto& s : wl.attention_spans) {
          unit_est += ceil_u64(s.q_tokens, qt) * wl.q_heads_local;
        }
        for (u64 rows : group_sides(unit_est * kp, ctx.hw.mesh_n)) {
          for (u64 cols : group_sides(std::max<u64>(1, unit_est * kp / rows),
                                      ctx.hw.mesh_n)) {
            const u64 G = rows * cols;
            if (G < kp || G > std::max<u64>(kp, unit_est * kp)) {
              continue;
            }
            for (u64 ctx_count : {1u, 2u}) {
              if (sram_attn(static_cast<double>(wl.elem_bytes), qt, kt,
                            wl.head_dim) *
                      static_cast<double>(ctx_count) >
                  ctx.limit) {
                continue;
              }
              if (diagnostics != nullptr) {
                ++diagnostics->attention_layout_evaluations;
              }
              if (!have_work) {
                work = aggregate_attention_work(wl, j.mode, qt, kt, kp,
                                                diagnostics);
                have_work = true;
              }
              Candidate c;
              build_attention(ctx, j, qt, kt, rows, cols, kp, ctx_count, work,
                              c);
              if (!c.stages.empty()) {
                out.push_back(std::move(c));
              }
            }
          }
        }
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// §5.4 fused FFN

struct FfnJob {
  const WorkloadQuantities* wl = nullptr;
  u64 layers = 1, passes = 1;
  u64 macro = 0;
};

double sram_ffn(double e, u64 m, u64 f, u64 H, u64 v) {
  // S_ffn = 6mH + 12mf + 4*max(kf, fn) at BF16 (§5.4): input e*mH, FP32
  // output partial 4mH, activation buffers 6e*mf, double-buffered weight
  // tile 2e*max(kf, fn).  The max(kf, fn) term is one shared slot, so the
  // feasible k and n chunk sizes coincide (k = n = v).
  return (e + 4.0) * static_cast<double>(m) * static_cast<double>(H) +
         6.0 * e * static_cast<double>(m) * static_cast<double>(f) +
         2.0 * e * static_cast<double>(v) * static_cast<double>(f);
}

void build_ffn(const Ctx& ctx, const FfnJob& j, u64 m, u64 f, u64 v, u64 rows,
               u64 cols, u64 ctx_count, Candidate& c) {
  const WorkloadQuantities& wl = *j.wl;
  const double e = static_cast<double>(wl.elem_bytes);
  const double L = static_cast<double>(j.layers);
  const double M = static_cast<double>(wl.m_tokens);
  const double H = static_cast<double>(wl.hidden);
  const double F = static_cast<double>(wl.ffn_local);
  const bool swiglu = wl.mlp_variant == MlpVariant::Swiglu;
  const double gemm_weight = swiglu ? 3.0 : 2.0;
  const double flops_w = swiglu ? 6.0 : 4.0;
  const double act_w = swiglu ? 16.0 : 15.0;
  const u64 T_m = ceil_u64(wl.m_tokens, m);
  const u64 T_f = ceil_u64(wl.ffn_local, f);
  const u64 sweeps_m = ceil_u64(T_m, rows);
  const u64 waves = sweeps_m * ceil_u64(T_f, cols);

  add_launch(c, ctx, j.macro, j.layers, j.passes);
  const size_t launch = 0;
  const u64 repeat = waves * j.layers;

  const double in_total =
      e * M * H * static_cast<double>(ceil_u64(T_f, cols)) * L;
  const double w_charged = gemm_weight * e * H * F *
                           static_cast<double>(std::max(j.passes, sweeps_m)) *
                           L;
  const size_t load = add_mem_stage(
      c, ctx, "load", j.macro, repeat,
      (in_total + w_charged) / static_cast<double>(repeat), true,
      ctx.noc.broadcast_unit(rows, cols), {{launch, DieStageDep::kAllDep}});

  const size_t pe = add_compute_stage(
      c, ctx, "ffn_pe", StageResource::PE, j.macro, repeat,
      flops_w * static_cast<double>(m) * H * static_cast<double>(f),
      {{load, 0}});

  // Activation eq-ops per (m, f) tile plus the F-distributed partial-sum
  // reductions across the columns (§5.4).
  const double act_vec = act_w * M * F;
  const double red_vec = static_cast<double>(cols > 1 ? cols - 1 : 0) * M * H;
  const size_t vec = add_compute_stage(
      c, ctx, "ffn_vec", StageResource::VEC, j.macro, repeat,
      (act_vec + red_vec) * L / (static_cast<double>(repeat) *
                                 static_cast<double>(rows * cols)),
      {{pe, 0}});
  c.eqops += (act_vec + red_vec) * L;

  size_t last = vec;
  if (cols > 1) {
    const auto unit = ctx.noc.merge_unit(rows, cols, cols);
    const double msg =
        static_cast<double>(cols - 1) * 4.0 * static_cast<double>(m) * H;
    add_compute_stage(c, ctx, "ffn_reduce_noc", StageResource::NOC, j.macro,
                      repeat, 0.0, {{pe, 0}}, unit, msg);
    last = c.stages.size() - 1;
  }

  const double out_total = e * M * H * L;
  add_mem_stage(c, ctx, "write", j.macro, repeat,
                out_total / static_cast<double>(repeat), true,
                ctx.noc.store_unit(rows, cols), {{last, 0}});

  c.flops = flops_w * M * H * F * L;
  c.summary.m_tile = m;
  c.summary.n_tile = f;
  c.summary.k_tile = v;
  c.summary.group_rows = rows;
  c.summary.group_cols = cols;
  c.summary.contexts = ctx_count;
  c.summary.fused = true;
  c.summary.layer_count = j.layers;
  c.summary.weight_passes = j.passes;
  c.summary.waves = waves;
  c.summary.weight_hbm_bytes = w_charged;
  c.summary.sram_peak_per_core =
      sram_ffn(e, m, f, wl.hidden, v) * static_cast<double>(ctx_count);
  c.summary.loop_kind = "ffn_fused";
}

std::vector<Candidate> map_ffn_fused(const Ctx& ctx, const FfnJob& j) {
  std::vector<Candidate> out;
  const WorkloadQuantities& wl = *j.wl;
  if (wl.ffn_local == 0 || wl.m_tokens == 0 || wl.hidden == 0) {
    return out;
  }
  const double e = static_cast<double>(wl.elem_bytes);
  for (u64 m : tiles_for(wl.m_tokens, kTokenTiles)) {
    const u64 T_m = ceil_u64(wl.m_tokens, m);
    for (u64 f : tiles_for(wl.ffn_local, kOutputTiles)) {
      const u64 T_f = ceil_u64(wl.ffn_local, f);
      for (u64 rows : group_sides(T_m, ctx.hw.mesh_n)) {
        for (u64 cols : group_sides(T_f, ctx.hw.mesh_n)) {
          for (u64 ctx_count : {1u, 2u}) {
            // Shared weight-chunk slot: largest feasible v (k = n = v).
            u64 v = 0;
            const std::vector<u64> vs = tiles_for(wl.hidden, kOutputTiles);
            for (auto it = vs.rbegin(); it != vs.rend(); ++it) {
              if (sram_ffn(e, m, f, wl.hidden, *it) * ctx_count <= ctx.limit) {
                v = *it;
                break;
              }
            }
            if (v == 0) {
              continue;
            }
            Candidate c;
            build_ffn(ctx, j, m, f, v, rows, cols, ctx_count, c);
            out.push_back(std::move(c));
          }
        }
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Trivial (zero-work) candidate: empty shard / fully-masked attention.

Candidate trivial_candidate(const Ctx& ctx, u64 macro, u64 layers,
                            u64 passes) {
  // Zero math and zero bytes, but the macro-group launches still happen
  // (§6.1; an empty shard is a legal dispatch, not a control marker).
  Candidate c;
  add_launch(c, ctx, macro, layers, passes);
  c.summary.layer_count = layers;
  c.summary.weight_passes = passes;
  c.summary.loop_kind = "trivial_zero_work";
  return c;
}

// Appends the stages of one sub-candidate (dropping its launch stage and
// rewiring dependencies) -- used by the FFN fallback composition.
void append_sub_candidate(Candidate& dst, const Candidate& sub,
                          size_t prefix) {
  for (const DieStage& s : sub.stages) {
    if (s.label == "launch") {
      continue;
    }
    DieStage t = s;
    for (DieStageDep& dep : t.deps) {
      dep.stage = dep.stage == 0
                      ? 0  // the single shared launch stage
                      : static_cast<std::uint32_t>(dep.stage + prefix);
    }
    dst.stages.push_back(std::move(t));
  }
  dst.flops += sub.flops;
  dst.eqops += sub.eqops;
  dst.matrix_bytes += sub.matrix_bytes;
  dst.vector_bytes += sub.vector_bytes;
  dst.matrix_service_units += sub.matrix_service_units;
  dst.vector_service_units += sub.vector_service_units;
  dst.noc_total_link_bytes += sub.noc_total_link_bytes;
  dst.noc_max_edge_bytes =
      std::max(dst.noc_max_edge_bytes, sub.noc_max_edge_bytes);
}

// (candidate selection with the shared §5.1 rule lives in CostModel.cc as
// detail::pick_best_candidate)

}  // namespace

// ---------------------------------------------------------------------------
// enumerate_candidates: dispatcher over op_kind (design doc §5)

std::vector<Candidate> enumerate_candidates(
    const WorkloadQuantities& wl, const Hardware& hw,
    EnumerationDiagnostics* diagnostics) {
  if (diagnostics != nullptr) {
    *diagnostics = EnumerationDiagnostics{};
  }
  NocRouter noc(hw);
  Ctx ctx{hw, noc, hw.sram_usable_bytes_per_core};
  std::vector<Candidate> out;

  // op_kind -> macro group table (frozen in the stage A2 report).
  u64 macro = 3;
  switch (wl.op_kind) {
    case OpKind::Norm:
      macro = 0;  // Norm1
      break;
    case OpKind::QkvRope:
      macro = 1;  // QKV (fused RoPE tail, no separate RoPE launch)
      break;
    case OpKind::Qk:
    case OpKind::ScaleMask:
    case OpKind::Softmax:
    case OpKind::Av:
    case OpKind::AttentionRegion:
      macro = 3;  // attention
      break;
    case OpKind::OProj:
      macro = 4;  // O
      break;
    case OpKind::Residual:
      macro = 5;  // res1
      break;
    case OpKind::MlpUp:
    case OpKind::MlpActivation:
    case OpKind::MlpDown:
    case OpKind::FfnRegion:
      macro = 7;  // FFN
      break;
    case OpKind::Logits:
      macro = 9;  // logits (outside the block's nine; charged the same way)
      break;
    case OpKind::LocalRegion:
      macro = 10;  // local
      break;
  }

  switch (wl.op_kind) {
    case OpKind::Norm: {
      if (wl.m_tokens == 0 || wl.hidden == 0) {
        out.push_back(trivial_candidate(ctx, macro, wl.layer_count, wl.weight_passes));
        break;
      }
      VectorJob j;
      j.rows_count = wl.m_tokens;
      j.row_len = wl.hidden;
      j.bytes_in_per_layer = wl.act_in_bytes_per_layer;
      j.bytes_out_per_layer = wl.act_out_bytes_per_layer;
      j.weight_row_bytes = wl.weight_bytes_per_pass;
      j.eqops_per_layer = wl.eqops_per_layer;
      j.layers = wl.layer_count;
      j.passes = wl.weight_passes;
      j.macro = macro;
      out = map_vector(ctx, j);
      break;
    }
    case OpKind::Residual:
    case OpKind::MlpActivation: {
      if (wl.eqops_per_layer <= 0.0) {
        out.push_back(trivial_candidate(ctx, macro, wl.layer_count, wl.weight_passes));
        break;
      }
      VectorJob j;
      j.rows_count = wl.m_tokens;
      j.row_len = wl.op_kind == OpKind::Residual
                      ? wl.hidden
                      : wl.ffn_local;
      j.bytes_in_per_layer = wl.act_in_bytes_per_layer;
      j.bytes_out_per_layer = wl.act_out_bytes_per_layer;
      j.eqops_per_layer = wl.eqops_per_layer;
      j.layers = wl.layer_count;
      j.passes = wl.weight_passes;
      j.macro = macro;
      out = map_vector(ctx, j);
      break;
    }
    case OpKind::ScaleMask:
    case OpKind::Softmax: {
      // Elementwise over materialized score rows; row length = the longest
      // span kv range (capacity), bytes/eq-ops from the workload floor.
      u64 kv_max = 0;
      for (const auto& s : wl.attention_spans) {
        kv_max = std::max(kv_max, s.kv_tokens);
      }
      if (wl.attention_pairs <= 0.0 || kv_max == 0) {
        out.push_back(trivial_candidate(ctx, macro, wl.layer_count, wl.weight_passes));
        break;
      }
      VectorJob j;
      j.rows_count = wl.m_tokens * wl.q_heads_local;
      j.row_len = kv_max;
      j.bytes_in_per_layer = 4.0 * wl.attention_pairs;  // FP32 scores
      j.bytes_out_per_layer = 4.0 * wl.attention_pairs;
      j.eqops_per_layer = wl.eqops_per_layer;
      j.layers = wl.layer_count;
      j.passes = wl.weight_passes;
      j.macro = macro;
      out = map_vector(ctx, j);
      break;
    }
    case OpKind::Qk:
    case OpKind::Av:
    case OpKind::AttentionRegion: {
      AttnJob j;
      j.wl = &wl;
      j.layers = wl.layer_count;
      j.passes = wl.weight_passes;
      j.macro = macro;
      j.mode = wl.op_kind == OpKind::Qk
                   ? AttnJob::Mode::QkOnly
                   : (wl.op_kind == OpKind::Av ? AttnJob::Mode::AvOnly
                                               : AttnJob::Mode::Full);
      out = map_attention(ctx, j, diagnostics);
      if (out.empty()) {
        out.push_back(trivial_candidate(ctx, macro, wl.layer_count, wl.weight_passes));
      }
      break;
    }
    case OpKind::QkvRope:
    case OpKind::OProj:
    case OpKind::MlpUp:
    case OpKind::MlpDown:
    case OpKind::Logits: {
      GemmJob j;
      const DieGemmShape& g0 =
          wl.gemms.empty() ? DieGemmShape{} : wl.gemms.front();
      if (!wl.gemms.empty()) {
        // The qkv_rope merged GEMM: [M,H]x[H, qh*d + 2*kvh*d] (three real
        // GEMMs share the M rows and input read, §4.2 rule 3).
        u64 n_total = 0;
        for (const DieGemmShape& g : wl.gemms) {
          n_total += g.n;
        }
        j.M = g0.m;
        j.K = g0.k;
        j.N = n_total;
      }
      j.elem = static_cast<double>(wl.elem_bytes);
      j.layers = wl.layer_count;
      j.passes = wl.weight_passes;
      j.macro = macro;
      if (wl.op_kind == OpKind::QkvRope && wl.position_encoding == PositionEncoding::Rope) {
        j.has_tail = true;
        j.tail_eqops = wl.eqops_per_layer;
        j.tail_bytes = wl.aux_bytes_per_layer;
      }
      if (j.M == 0 || j.K == 0 || j.N == 0) {
        out.push_back(trivial_candidate(ctx, macro, wl.layer_count, wl.weight_passes));
        break;
      }
      out = map_gemm(ctx, j);
      break;
    }
    case OpKind::FfnRegion: {
      FfnJob fj;
      fj.wl = &wl;
      fj.layers = wl.layer_count;
      fj.passes = wl.weight_passes;
      fj.macro = macro;
      std::vector<Candidate> fused = map_ffn_fused(ctx, fj);
      // §5.4 "无可行融合映射则退回分离算子": the separate-ops fallback is
      // built only when no fused mapping is feasible; its intermediates are
      // materialized with their real external traffic.
      const bool swiglu = wl.mlp_variant == MlpVariant::Swiglu;
      GemmJob up;
      up.M = wl.m_tokens;
      up.K = wl.hidden;
      up.N = swiglu ? 2 * wl.ffn_local : wl.ffn_local;
      up.elem = static_cast<double>(wl.elem_bytes);
      up.layers = wl.layer_count;
      up.passes = wl.weight_passes;
      up.macro = macro;
      GemmJob down;
      down.M = wl.m_tokens;
      down.K = wl.ffn_local;
      down.N = wl.hidden;
      down.elem = static_cast<double>(wl.elem_bytes);
      down.layers = wl.layer_count;
      down.passes = wl.weight_passes;
      down.macro = macro;
      VectorJob act;
      act.rows_count = wl.m_tokens;
      act.row_len = wl.ffn_local;
      act.bytes_in_per_layer = wl.act_in_bytes_per_layer;
      act.bytes_out_per_layer = wl.act_out_bytes_per_layer;
      act.eqops_per_layer = wl.eqops_per_layer;
      act.layers = wl.layer_count;
      act.passes = wl.weight_passes;
      act.macro = macro;
      Candidate fallback;
      bool have_fallback = false;
      if (fused.empty() && up.N > 0 && down.K > 0 && act.row_len > 0) {
        std::vector<Candidate> ups = map_gemm(ctx, up);
        std::vector<Candidate> acts = map_vector(ctx, act);
        std::vector<Candidate> downs = map_gemm(ctx, down);
        if (!ups.empty() && !acts.empty() && !downs.empty()) {
          const Candidate& best_up = ups[pick_best_candidate(ups, hw)];
          const Candidate& best_act = acts[pick_best_candidate(acts, hw)];
          const Candidate& best_down = downs[pick_best_candidate(downs, hw)];
          add_launch(fallback, ctx, macro, wl.layer_count, wl.weight_passes);
          append_sub_candidate(fallback, best_up, 0);
          append_sub_candidate(fallback, best_act, fallback.stages.size() - 1);
          append_sub_candidate(fallback, best_down, fallback.stages.size() - 1);
          fallback.summary = best_up.summary;
          fallback.summary.loop_kind = "ffn_fallback_separate";
          fallback.summary.fused = false;
          fallback.summary.fallback = true;
          fallback.summary.sram_peak_per_core =
              std::max(std::max(best_up.summary.sram_peak_per_core,
                                best_act.summary.sram_peak_per_core),
                       best_down.summary.sram_peak_per_core);
          have_fallback = true;
        }
      }
      if (fused.empty() && !have_fallback) {
        out.push_back(trivial_candidate(ctx, macro, wl.layer_count, wl.weight_passes));
        break;
      }
      out = std::move(fused);
      if (have_fallback) {
        out.push_back(std::move(fallback));
      }
      break;
    }
    case OpKind::LocalRegion: {
      out.push_back(trivial_candidate(ctx, macro, wl.layer_count, wl.weight_passes));
      break;
    }
  }

  for (Candidate& c : out) {
    c.summary.op_kind = wl.op_kind;
    c.summary.profile = hw.profile;
  }
  if (out.empty()) {
    throw DieModelError("enumerate_candidates: no feasible mapping for op_kind " +
                        std::string(to_string(wl.op_kind)) +
                        " on this hardware (per-core SRAM budget " +
                        std::to_string(hw.sram_usable_bytes_per_core) +
                        " bytes)");
  }
  return out;
}

std::vector<Candidate> enumerate_candidates(const WorkloadQuantities& wl,
                                            const Hardware& hw) {
  return enumerate_candidates(wl, hw, nullptr);
}

}  // namespace detail
}  // namespace DieModel
