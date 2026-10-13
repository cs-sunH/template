# Vector budget and parameter-residency audit (scratch)

This is a read-only audit of the current design text and `die/src/Mapping.cc`. It is not a compiler change, a timing calibration, or a granularity pass. No tracked files were changed.

## Source contract and discrepancy

The design's §6.4 (lines 366–372) assigns vector work to contiguous row/tile shards in tile-id order, with `core = tile_id mod N_core`. It charges gamma/beta and cos/sin the first time each consuming Core reads them. A parameter slice can be reused within the region only if that complete slice and the other live buffers fit SRAM at once; reuse across regions is absent unless task residency proves it. §6.4 asks for separate `math_vec_ops`, `vec_equivalent_ops`, `special_function_counts`, and `conversion_elements`, and says target quantization and accumulator conversion are charged by actual converted counts.

Current design §7.2 (lines 448–458) explicitly uses `active_cores` in the COMPUTE_GATE denominators and retains the GPU-slot conservative occupancy model. The parent confirmed this is the current profile contract. A full-die `/N_core` denominator is a separate research version and is not the current design. The helper/audit formulas below follow active-core conservative occupancy and replace only average wave work with a slowest-Core cohort sum.

Current mapping evidence:

- `compute_time()` divides PE and Vec work by `active`, lines 607–616. `seal_activity()` copies the resulting `compute_s` to both `private_time_s` and `work[COMPUTE_GATE]`, lines 619–640. `Flow` has only one compute-time scalar. Consequently the global gate and private limit cannot express different denominators or tail imbalance.
- `add_vector()` bills `input_bytes + extra_input` once, schedules `eqops` unchanged, and uses the same `compute_s` for both bounds, lines 886–906. It does not add dtype conversion counts.
- The vector SRAM peak is built from at-most-256-element tiles. Norm replaces `tile_extra` with one or two parameter chunks of `min(H,256)` elements, rather than testing the full H-length parameter slice, lines 907–922. Rounds flatten `elements` into `active*256` work, lines 923–930.
- Operator construction selects `active=min(N_core,M)` for every vector path, lines 1517–1524. Norm bills one H-length parameter tensor (RMSNorm) or two (LayerNorm) once, lines 1525–1532. RoPE bills its full angle table once, with `rope_count=M*(q_heads+kv_heads)*head_dim*executions`, lines 1551–1573. `conversion_elements` is not updated for these vector primitives; `Metrics` has no special-function count field and `finalize_summary()` does not emit those requested fields.

## Shared gate and PRIVATE split

Let `C=N_core`; let `P_pe`, `P_vec` be per-Core peaks; and let `η_pe`, `η_vec` be efficiencies. For a fixed ordered list of consumer tiles, make cohorts `g` from consecutive groups of at most C tiles. A tile's serial PE+Vec budget is

```text
p_compute(i) = I_i/(P_pe*η_pe) + V_i/(P_vec*η_vec)
```

For each cohort, `active_g` is its actual number of tiles. Balanced tile work reproduces §7.2's active-core average `Σ_i I_i/(active_g*P_pe*η_pe) + Σ_i V_i/(active_g*P_vec*η_vec)`; to account for tails, use the slowest-Core extension `Σ_g max_i_in_g p_compute(i)`. For strict FP32 Vec matrix work, matrix ops are part of `V` and `I=0`. `repeat_count` remains audit metadata; work totals already include actual repeats.

A stage's local bounds are its actual per-Core LOCAL, SRAM_READ, and SRAM_WRITE bytes divided by the respective one-Core service rates, then maximized with its compute bound. The stage-specific byte terms must use bytes consumed in that activity; they should not include HBM bytes a separate load/store activity owns. This is a capacity lower bound, not a sum of compute and memory times.

Keep the shared COMPUTE_GATE and `private_time_s` fields independently derived, but do not substitute `/N_core` as the current shared-gate denominator. Under the current conservative profile the compute-gate work itself follows active-core cohort occupancy; `private_time_s` then also applies the stage's one-Core SRAM/LOCAL caps. `add_flow`, scaling, epoch division and first/interior/last-wave construction must preserve the cohort-specific values, not replace non-uniform tails with an average. A repeated activity needs the per-wave bound for its repeated wave and separate first/last/tail activities where needed.

## Consumer mapping and math work

Use `X=layer_instances*weight_passes` only where the compiled region actually repeats; default parameter reuse does not cross distinct regions/layers. Each fixed template must name its consumer tile and owner before deriving PRIVATE or parameter fanout.

| Primitive | Consumer unit that preserves semantics | Base math Vec equivalent ops | Parameter source |
|---|---|---:|---|
| RMSNorm / LayerNorm | A complete normalization row (or an explicitly versioned row tile); one row has a reduction over H and cannot be striped over cores without a reduction template | `6*M*H*X` / `8*M*H*X` | gamma H / gamma+beta 2H |
| RoPE after QKV | A declared `(token, head-slice)` tile; cosine/sine are shared by all heads that a given Core consumes for that token | `3*M*(q_heads+kv_heads)*head_dim*X` | 2 angle vectors of length `M*head_dim` |
| Residual, activation, scale/mask | Contiguous independent element tiles; no cross-element reduction | Residual `E`; SwiGLU `16E`; GELU `15E`; scale/mask `2*score_slots` | None; residual's second tensor and SwiGLU's second tensor are data operands, not parameters |
| Softmax | One `(batch, query-head, query-row)` tile, since max/sum reductions span that row; do not split a row across Cores without an explicit reduction primitive | `10*processed_score_slots` | None |
| Q/K/V/O dense projection | Its GEMM mapping, not the vector tile path | GEMM-specific | Weights are GEMM inputs |

For elementwise independent tensors a natural bounded template uses contiguous tiles of `V_tile` elements (the current code happens to use 256 as a cap, but §6.4 does not standardize that number): `T=ceil(E/V_tile)`, actual tile sizes `e_i`, owner `i mod C`, and `active=min(C,T)`. For Norm and Softmax, row identity takes precedence over this flat mapping. For RoPE, specify whether the core owner is a row or a `(row,head-slice)`; only then can parameter fanout be counted.

For an elementwise vector tile i, after defining its conversion roles, use

```text
V_i = α*e_i + Σ_r κ_r * converted_elements(i,r)
W_compute_gate = Σ_g max_{i in cohort g} V_i/(P_vec*η_vec)
W_private_compute = W_compute_gate
```

Each cohort is the next at-most-C tiles in the declared tile-id order, where `C=N_core`; its final partial cohort has its actual smaller active count. For uniform work, a cohort maximum equals the §7.2 `Σ_i V_i/(active_cores*P_vec*η_vec)` wave budget. The maximum is the specified conservative tail correction. Do not retain the flat `Σ_i/(C*P_vec*η_vec)` average as the current gate formula; a separate model could study that approximation if it is clearly versioned. `PRIVATE` then applies its existing activity-level LOCAL/SRAM bounds to this compute term; those independent byte terms are not folded into the shared gate formula here.

For a mixed PE/Vec tile, add the serial PE term to each tile before taking the cohort maximum. `α` is the fixed operation template above; `κ_r` is the target cost of each actual conversion and remains a nonnegative real coefficient (including fractional values). For a BF16/INT32→FP32 conversion, a fixed template may price each converted element at 1 equivalent op; conversion to INT4 uses `κ_quant`; native execution dtype costs zero. Output casts have their own declared coefficient. The `6/8/3/...` math template must not be multiplied again merely because a conversion is added.

For each vector role, independently record: source bytes, converted element count, converted-buffer bytes if materialized, SRAM read/write bytes, and final HBM output bytes. HBM parameter traffic can be reduced by an SRAM cache, but repeated SRAM accesses remain LOCAL/SRAM work. INT4 row/tile payloads must use the actual packing boundary; for independent packed rows use `M*ceil(H/2)` bytes, not automatically `ceil(M*H/2)`.

## Norm parameter residency and long-row bound

Let `P=1` for RMSNorm and `P=2` for LayerNorm, `A=min(C,M)` row-owning consumer Cores, and `b_p` the source parameter element width. Treat gamma and beta as separate length-H vectors. If each is INT4-packed independently, the parameter payload is `P*ceil(H/2)` bytes per owner; otherwise it is `P*H*b_p`.

Define the resident parameter slice for a Core from the complete parameters it will use, plus all simultaneously live input, converted data, reduction state, output staging and metadata. The reuse check is

```text
B_other + B_parameter_slice <= floor(rho_sram * SRAM_bytes_per_core)
```

The template must say whether raw and converted parameter buffers coexist; if conversion is in-place, count only the converted resident buffer after the source read. No partial-slice reuse is implied by §6.4. If complete gamma/beta plus other live state fails this test, do not credit across-row parameter reuse.

A bounded row-oriented execution can separate two decisions:

1. If a complete row's input and required streaming/reduction state fit, keep that row on one Core. It can read the input once even when the full parameter vector does not fit; in that case stream the parameters once per row. If the full parameter slice also fits, each consumer Core loads its complete slice once and reuses it over its assigned rows.
2. If the input row itself does not fit the declared row buffer, use an explicit two-pass algorithm: pass 1 reads X to accumulate row statistics; pass 2 rereads X, applies normalization and emits output. The parameter slice is either already cached or read for the output pass. Do not quietly charge one input pass for this long-row template.

With `N_input_passes=1` or 2 and `N_parameter_loads=A` when a complete parameter slice fits (otherwise M row loads), a minimum physical HBM ledger is

```text
D_X = X * N_input_passes * M * bytes_per_row(H, activation_dtype)
D_parameter = X * N_parameter_loads * P * bytes_per_row(H, parameter_dtype)
D_Y = X * M * bytes_per_row(H, output_dtype)
```

For INT4 rows/vectors, `bytes_per_row(H,int4)=ceil(H/2)` per packing boundary. The HBM parameter read count is per Core when cached, but parameter SRAM/LOCAL reads still follow use: normally `X*M*P*H*bytes_per_execution_parameter` for gamma/beta consumed once per output element. If uncached, its HBM traffic rises to per-row while LOCAL/SRAM use remains per-row. If input conversion is required, conversion elements follow actual passes (`X*N_input_passes*M*H`); cached parameter conversion follows `X*A*P*H`, uncached parameter conversion follows `X*M*P*H`, and output conversion follows `X*M*H`. Corresponding real `κ` values enter Vec work. This accounting is separate from the reference math template, which remains `6` or `8` ops/element.

The exact fit buffer `B_other` and exact reduction/output staging must be part of a fixed template. The current source does not establish them: it sizes only one 256-element parameter chunk, leaves input pass count at one, and does not state a row-reduction algorithm. The bounded two-pass recommendation is a template to specify, not evidence that the current 6/8 eq-op total already includes a particular pass schedule. The base 6/8 math ops should remain once per element unless the versioned implementation shows extra arithmetic; rereading/converting input is additional byte/conversion work, not automatically another full `6/8` charge.

Example to make the replication visible (BF16, H=4096, M=8, C=4, one invocation): RMS gamma is 8 KiB per owner, so a fitting cache reads 32 KiB HBM across four consumers; no cross-row cache reads 64 KiB (one 8 KiB vector per row). LayerNorm has two vectors: 64 KiB cached across four owners or 128 KiB per-row without reuse. Input is 64 KiB for one pass or 128 KiB for two; output is 64 KiB. These are accounting examples; no capacity-fit conclusion is possible without the live-buffer template and hardware SRAM value.

## RoPE angle ownership

Let `R=M*(q_heads+kv_heads)*head_dim*X`. Base math is `3R` eq-ops. For each token position m, let `K_m` be the set of consumer Cores that own at least one Q/K head-slice using that position. Reuse is allowed only if each Core's complete cos/sin parameter slice for its assigned positions, together with that Core's other live buffers, fits SRAM. If it fits, physical angle HBM reads are

```text
D_angles = X * Σ_m |K_m| * 2*head_dim * bytes(parameter_dtype)
```

and angle conversion elements are `X*Σ_m |K_m|*2*head_dim` when the parameter dtype differs from execution dtype. If that complete per-Core slice does not fit, do not claim cross-tile/region cache reuse: replace `|K_m|` by the actual number of Core/tile angle-slice loads. This charges no free broadcast while allowing same-Core reuse only under the full-slice fit test. Q/K input conversion and output conversion are each based on their actual `R` element visits; V is untouched. The QKV projection remains a preceding GEMM. RoPE's Q/K materialization, parameter reads, SRAM/LOCAL reads, and output writes must be separately identified.

The current row-active choice could support `|K_m|=1` if a row owns all heads, in which case the current global `2*M*head_dim` table size happens to match one angle read per token. But current flattened 256-element rounds do not encode that ownership, and there is no per-consumer fit/replication check. A head-sliced template would need larger parameter traffic. Do not multiply by `active` without first fixing the ownership rule.

## Dense vector and score-row private accounting

For elementwise residual/activation/scale operations, set tile math sizes from real tail elements. A cohort private bound is the largest tile cost in that cohort, summed across cohorts. `active=min(C,M)` is not the natural limit for an independent elementwise tensor; `min(C,ceil(E/V_tile))` may use more Cores when M is small and H/F is large. The current implementation's `M`-based active limit can be retained as a conservative policy only if declared; it is not the general row/tile rule.

For softmax, identify rows from `(span,batch,query-head,query-index)` and assign each row to exactly one Core. The tile's `e_i` is the actual set of score slots processed for that row (including explicitly masked dense slots where the template performs them). Then `W_private=Σ_g max_i 10*e_i/(P_vec*η_vec)` before conversions and local-resource bounds. `scale_mask` is elementwise absent a reduction; a fused scale+softmax region should keep the same score-row owner so no hidden cross-Core reduction is introduced. Current `active=min(C,M)` and `ceil(score_slots/(active*256))` do not establish query-head row ownership or a reduction boundary.

## Small two-COMP split fixture

A minimal arithmetic fixture distinguishes average and slowest-cohort accounting without changing the current gate denominator:

- Hardware: `N_core=8`, `P_vec_per_core*eta_vec=1 op/s`, COMPUTE_GATE capacity `1 s/s`; no HBM/LOCAL/SRAM limit.
- Each COMP request maps three ordered tiles to two active Cores, with Vec costs `[2, 0.5, 1]` eq-ops. Cohorts are `[2,0.5]` and `[1]`, so `active_last=1`, `last_max=1 s`, and the serial slowest-cohort gate is `2+1=3 s`.
- The old whole-task average `total_ops/active=3.5/2=1.75 s` misses the cohort barrier. Two concurrent requests with the corrected current-profile work of 3 s each are gate-limited to progress 1/6 each and finish in 6 s; the average-based value would predict 3.5 s each under equal sharing. This is a conservative GPU-slot model example, not a physical disjoint-Core claim.

A Runtime fixture can use 8-Core synthetic hardware and two small plans with the explicit tile-work ledger `[2,0.5,1]`; make HBM/LOCAL/SRAM nonbinding. Report `active_last`, `last_max`, serial slowest compute-gate work, actual Runtime finish and gate utilization separately. Do not assert that the aggregate runtime tracks physical Core ownership.

## Findings and limits

The specification requires parameter-byte fanout at consumer-Core granularity, full parameter-slice fit before reuse, distinct math/conversion counts, and the active-core conservative compute gate. Current vector mapping has no closed accounting for these. Norm's one-pass/global-parameter assumption and flat tile schedule are not enough to certify large rows. RoPE ownership must be made explicit before its parameter replication can be charged. Generic elementwise tiles and reduction rows need different consumer maps. The scratch formulas are directly implementable once the fixed buffer lifetimes, execution dtype, tile owner, and conversion coefficients are selected; this audit does not choose a microkernel or claim those choices are validated.
