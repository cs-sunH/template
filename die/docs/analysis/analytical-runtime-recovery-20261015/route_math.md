# OS mapped-route cumulative math (scratch only)

This note is a read-only derivation for the production discussion. It does not
change `die/` code or formal tests. The bounded target is exact aggregate
HBM/NoC work by mapped OS destination, with no loop over output tiles or cores.

## Existing pieces to keep

- `GemmOSGeometry::AccessClass` (at most 12 in practice, bounded by
  `kMaxAccessClasses`) carries an M/N/K shape class and the exact integer
  `per_consumption` ledger returned by `GemmOSAccessCounter::count`.
- `class_output_count_in_range` and private `prefix_count` already count the
  class's row-major output q values in a range with a fixed number of ops.
- `class_consumption_count_in_cohort`, `class_output_count_in_wave`,
  `wave_ledger`, `cohort`, `output_fact` define the exact first-wave,
  first-cohort, final-wave and all-work populations. `output_fact` confirms
  the destination lane is `q % core_count`.
- `destination`, `point_abs_axis`, `uniform_source_mean_abs`, and `route_bytes`
  in `Mapping.cc` are useful topology/route formulas, but `AxisDist` currently
  assumes every active lane carries equal bytes. Exact OS routing should keep
  their topology meanings while replacing the unweighted average with
  class-weighted destination summaries.

## Fixed-size q decomposition

For one M/N class, q is `m * Nt + n`, with M and N tile-index ranges from the
axis classes. N classes have only two forms: the full-N prefix
`n in [0,F)` and, if present, the one N-tail column `n=F`. This means a class
intersected with any q interval `[a,b)` is a fixed-size union of:

1. one or two partial row intervals;
2. a middle interval of complete rows; and
3. for a full-N class when `F < Nt`, subtraction of the N-tail AP
   `q = m*Nt + F` over those complete rows.

If N has no tail, the full-N portion is simply one contiguous q interval. An
N-tail class itself is one arithmetic progression (AP). Intersecting the
AP with `[a,b)` only clips its m-index interval. Therefore all route counts
for one AccessClass need a constant number of interval/AP terms, not a loop
over M, N, K, tiles, rounds, or cores. The existing `prefix_count` remains the
count-only special case.

For a term `q_i = b + d*i`, `0 <= i < n`, physical lane is `r_i=q_i mod C`,
where `C=core_count=nx*ny` for validated hardware. A standard Euclidean
`floor_sum` is

```text
FS(n,m,a,b) = sum(i=0..n-1) floor((a*i+b)/m).
```

For `0 <= t <= C`, the number of AP destinations whose lane is below `t` is

```text
Nlt(t) = n - FS(n,C,d,b+C-t) + FS(n,C,d,b).
count_lane_interval(l,h) = Nlt(h) - Nlt(l).
```

This follows because
`floor((q+C-t)/C)-floor(q/C)` is one exactly when `q mod C >= t`.
It handles wraparound and non-coprime AP strides without enumerating residues.
Use checked `u128` arithmetic and the standard logarithmic Euclidean
floor-sum reduction; AP construction/row clipping is bounded by the shape
class count.

## Byte-weighted route summary

For class `i`, let `b_i` be one `per_consumption.hbm_read_bytes` or
`hbm_write_bytes`; let `k_i` be the K multiplicity selected for this flow
(all K tiles, one first/final wave, or the first cohort), and let `E` be the
execution repeat count. The exact class payload attached to lane `r` is

```text
B_i(r) = b_i * k_i * E * count_i(q in selected range, q mod C = r).
```

No per-lane vector is needed. The AP floor-sum can provide byte-weighted
statistics directly by summing lane-coordinate functions. For lane `r`,
`x=r mod nx`, `y=floor(r/nx)`. `Mapping.cc` only needs a fixed set of
statistics for each row and class:

- total bytes;
- bytes with x left/right of `cx`, and y low/high of `cy` (directional
  XPOS/XNEG/YPOS/YNEG work);
- byte-weighted sums of the topology's expected absolute x/y distance.

For a fixed source x, the latter is a weighted sum of `|x-s|`; explicit
ingress uses the existing finite `source_x[row]` list. For a uniform source
axis of length D, use the per-destination closed form

```text
g_D(z) = (z*(z+1) + (D-1-z)*(D-z)) / (2*D).
```

It is quadratic in z, so the residue reducer needs count/first/second moments
(plus prefix moments at the finite explicit source columns). A small
`floor_sum_moments` helper can accumulate those moments in the same
Euclidean recursion; it must be logarithmic in C and return no per-core
storage. For y, because `C=nx*ny`, `y = floor((q mod C)/nx)`; a quotient/remainder
moment helper for this nested modulus is needed. The top-level API should
hide that arithmetic and return route moments, rather than exposing a
callback that invites a core loop.

One suitable minimal internal API is:

```cpp
struct MappedRouteMoments {
  u128 bytes;
  u128 x_left_bytes, x_right_bytes, y_low_bytes, y_high_bytes;
  long double x_abs_byte_sum, y_abs_byte_sum;
};
MappedRouteMoments GemmOSGeometry::class_route_moments(
    size_t class_index, u128 q_begin, u128 q_end,
    u64 k_begin, u64 k_count, u64 nx, u64 ny,
    const SourceAxis& source_x, u64 source_y);
```

The name/signature is illustrative. Prefer putting generic q/AP lane-moment
math next to `GemmOSGeometry`, while `Mapping.cc` applies the topology source
distribution. A still smaller boundary is `class_lane_coordinate_moments`
that returns exact count and x/y first/second/prefix moments; it must not
return a `vector<per_core>`.

For one row, given source-left probability `pL` and source distance moment
`H`, route work is then formed from the weighted destination totals, e.g. on
reads `XPOS=pL*bytes(x>=cx)`, `XNEG=(1-pL)*bytes(x<cx)`; writes reverse source
and destination sides. `HOP=x_abs_byte_sum+y_abs_byte_sum`. Injection work
uses the exact row byte total. This reproduces the current route definitions
while weighting each shape class by its actual q-to-core mapping.

For cumulative service primitives, request the corresponding q ranges using
the already existing wave/cohort facts: first load is wave 0, first cohort
uses `class_consumption_count_in_cohort(0, i)`, total uses each class's
`repeat_count`, and final store uses the last wave. The subsequent
`prefix_load`, `remaining_load`, and `early_store` remain differences of
conservatively apportioned aggregate flows. This keeps the number of curve
segments fixed.

## Odd two-row HBM split

Never independently apply `ceil(class_bytes/2)` to every class: for several
odd class totals, their row-0 sum exceeds `ceil(total/2)`. Define one stable
class/order prefix P per HBM direction and apportion every selected class
block `[P,P+B)` by prefix differences:

```text
row0(P,B) = ceil((P+B)/2) - ceil(P/2)
row1(P,B) = floor((P+B)/2) - floor(P/2).
```

Then `sum(row0)=ceil(total/2)` and `sum(row1)=floor(total/2)` exactly, for any
class partition. A class-total prefix alone does **not** define additive row
bytes for the non-contiguous first-wave / first-cohort / final-wave subsets
used by the service curve. Those subsets need either a canonical event-level
byte order whose odd-byte parity can be counted analytically, or one shared
class-resolved ledger from which every subflow is selected consistently. Do
not assume independently split subset totals will subtract safely. The
existing model specifies only an aggregate equal split; it does not specify
which physical byte of a tile lives in which HBM row. This is a conservation
policy for aggregate row bytes, not evidence of a physical per-byte row
address assignment.

If later work needs each service subflow's row payload, derive an additive
event-level partition and verify it separately. This note does not yet prove
how to count that parity for all interleaved q/K/execution subsets.

## Scope and failure mode to avoid

Current OS `add_hbm(load, read_total, active, ...)` calls route one scalar
payload through `destination(active,...)`, which assumes uniform bytes over a
row-major active prefix. The real OS ledger says that tile q belongs to
`q % ncore`, and its bytes vary with M/N tails and K role. It can therefore
misweight hop/directional NoC work even when the total HBM/SRAM ledger is
correct. The fix is bounded per AccessClass route moments, not a per-Core
histogram and not a changed Runtime/credit model.

WR is a distinct continuation: reuse the AP/floor-sum machinery only after
writing down its actual output-slot-to-core ordering; do not inherit OS
`q % C` mapping by assumption. Other templates can use their own finite
shape/access classes and the same route-moment reducer when their mapping is
specified.
