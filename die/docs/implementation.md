# Coarse Die v2 implementation and evidence

> This is an implementation description, not an acceptance certificate. [HANDOFF.md](HANDOFF.md) is the current progress and defect ledger. Described intentions below must not be read as passed features.

This page describes the files currently under `die/`, including what the implementation consumes and what its reported results mean. The v2 resolver and runtime use the fixed source revision in `configs/example_hardware.json`; changing the architecture source table, task template, event ordering, or rounding contract requires a version change and corresponding evidence.

## Build and verification commands

Build the canonical standalone library from the repository root. `COARSE_DIE_JSON_INCLUDE` must name the parent directory containing `json/json.hpp`:

```bash
source /workspace/cloud-setup/activate.sh
cmake -S die -B build/coarse_die_v2 \
  -DCOARSE_DIE_JSON_INCLUDE="$PWD/astra-sim-wscllm/extern/helper"
cmake --build build/coarse_die_v2 -j
ctest --test-dir build/coarse_die_v2 --output-on-failure
./die/tools/estimate.py \
  --library build/coarse_die_v2/lib/libcoarse_die_cabi.so \
  --hardware die/configs/example_hardware.json \
  --tasks die/configs/example_tasks.json \
  --output -
```

The static target is `coarse_die`; the shared C ABI target is `coarse_die_cabi`, emitted at `build/coarse_die_v2/lib/libcoarse_die_cabi.so`. Tests are enabled by default. CMake registers whichever of `source_golden_test`, `hardware_search_test`, `compile_execution_test`, `runtime_semantics_test`, `oracle_granularity_test`, and `capi_lifecycle_test` exist in the source snapshot; the benchmark is a separate `coarse_die_benchmark` target. A CMake configure that silently skips an expected test is not evidence that the skipped contract passed.

`tools/estimate.py` is the user-facing offline path through the public ABI. `--tasks` accepts a single task object, a task array, or `{"specs":[...]}`. It prints one result/status per input item, writes to stdout by default (`--output -`) or to the selected path, returns nonzero when an item fails, and frees both the result buffer and model handle on every exit path. It does not compute timing formulas in Python.

The source-level evidence commands are:

```bash
./die/tools/calibrate_tx8.py
python3 die/tools/sync_repositories.py --check
```

The first verifies the workbook cache and reference reconstruction. The second checks vendored release contents, once the release has been copied. Neither command runs the v2 C++ runtime or validates hardware performance.

## Hardware input contract and field consumers

The descriptor is a JSON object with `schema: "coarse_die_v2"`, the immutable `source_revision`, and required `architecture`, `search_profile`, and `performance_profile` objects. `core_resource_table` and `source_constants` are optional mirrors; if supplied, they must match the compiled source exactly. The canonical resolved descriptor includes these mirrors and a content fingerprint.

| Input field | Accepted values / units | Consumer and effect |
|---|---|---|
| `source_revision` | Exact revision in `configs/example_hardware.json` | Rejects a descriptor from a different architecture source revision. |
| `core_resource_table.id`, `.rows_sha256` | `core64_v2` and the compiled 64-row digest | Identifies/checks the source-derived `(core_type, SRAM, NoC width) → N_vec, N_pe` table. |
| `source_constants` | Full exact mirror when present | Checks the assumptions recorded by the source; it does not expose run-time free parameters. The complete current set is in `configs/example_hardware.json`. |
| `architecture.core_type` | `small` or `large` | Selects Core-edge formula and one of the two source table families. |
| `architecture.sram_mib_per_core` | One of 0.25–2.00 MiB in 0.25 MiB steps | Selects the Core table row, SRAM capacity, and Core L0 edge. MiB uses 2²⁰ bytes. |
| `architecture.noc_width_bits` | 64, 128, 256, or 512 | Selects the source table row and per-link/local-bus width used by resource-rate formulas. |
| `architecture.core_columns_nx`, `core_rows_ny` | Positive integer counts | Set the rectangular Core grid, core count, mesh directions, edge count, and minimum Die outline. Core assignment and routing profiles use this shape. |
| `architecture.core_pitch_x_mm`, `core_pitch_y_mm` | Non-negative mm | Add inter-Core spacing to the minimum width and height. |
| `architecture.die_extra_width_mm`, `die_extra_height_mm` | Non-negative mm | Add the declared peripheral allowance to the minimum grid dimensions. |
| `architecture.die_geometry` | `minimum_grid`, `grow_width`, `grow_height`, `isotropic`, or `resolved_source_boundary` | Chooses the identified source geometry boundary and validates it against the derived minimum grid / required area. |
| `architecture.geometry_candidate`, `upstream_search_profile` | Required for `resolved_source_boundary` | Record which permitted boundary and upstream search profile produced a resolved outline. |
| `architecture.die_width_mm`, `die_height_mm` | Optional positive mm | If given, must equal the identified source-boundary geometry; they are checked, not arbitrary tuning knobs. |
| `architecture.hbm_stacks_per_row` | 1–14 | Sets two-row HBM capacity and raw row bandwidth; each row has `n` stacks. |
| `architecture.hbm_placement_mode` | `same_side_two_rows` or `two_sides_one_row_each` | Controls row-to-ingress mapping for bounded ingress profiles; it does not change HBM capacity or raw bandwidth. |
| `architecture.system_region_width_mm`, `system_region_height_mm` | Positive mm | Determine the number of source-derived modules that fit in the system region and system capacity/peak totals. |
| `search_profile.die_area_min_mm2`, `die_area_max_mm2` | Positive mm² | Filter the Die outline and determine the minimum source-area requirement. |
| `search_profile.reticle_width_mm`, `reticle_height_mm`, `allow_reticle_rotation`, `die_max_aspect_ratio` | mm / boolean / ratio ≥ 1 | Filter candidates against reticle bounds and maximum aspect ratio. |
| `search_profile.require_nonnegative_d2d_budget` | Boolean | Rejects a Die whose bump-area budget minus two-row HBM peak is negative. |
| `search_profile.require_hbm_row_within_die` | Optional boolean, default false | Adds an HBM row width fit check. |
| `search_profile.geometry_search_mode` | `core_first` or `wafer_first` | Selects the source search procedure. `wafer_first` also consumes wafer column/row ranges. |
| `search_profile.area_extension_rules` | Any unique subset of `grow_width`, `grow_height`, `isotropic` | Bounds which area-extension geometries the search can try. |
| `search_profile.nx_range`, `ny_range`, `hbm_n_range` | `{min,max}` or two-element integer range | Set finite search candidates for grid dimensions and stacks per row. |
| `search_profile.wafer_columns_range`, `wafer_rows_range` | Required for `wafer_first` | Bound the selected module columns and rows after geometry filtering. |
| `search_profile.objective` | `compute_peak`, `hbm_capacity`, or `pareto` | Sorts feasible hardware candidates by source-derived aggregate compute, capacity, and D2D tie-breaks. It is not a performance fit. |
| `performance_profile.execution_profile`, `calibration_status` | Version/profile labels | Record the assumed compute templates and evidence status. `assumed` is not promoted by a successful estimate. |
| `performance_profile.timing_topology_profile` | Required explicit choice: `distributed_ideal_v2`, `edge_single_word_v2`, `explicit_ingress_v2` | Chooses whether aggregate HBM ingress is ideal/unlimited, edge-word bounded, or supplied explicitly. No choice is inferred as physical truth. |
| `performance_profile.routing_profile` | `mapped_uniform_sources_v2`, `uniform_grid_v2`, `edge_uniform_v2`, `uniform_proxy_v2` | Chooses the declared source/destination distribution model for route work. |
| `performance_profile.pipeline_profile` | Current template: `pipeline_fluid_v2` | Names the fluid overlap model; event-level granularity remains a separate validation status. |
| `performance_profile.granularity_validation`, `physical_topology_validation` | Status strings | Preserve the validation state. Current example uses `not_run` and `unverified`. |
| `eta_pe`, `eta_vec`, `eta_hbm_matrix`, `eta_hbm_vector`, `eta_sram`, `eta_local`, `eta_mesh`, `eta_inject` | `(0,1]` efficiencies | Scale the explicitly named PE, Vec, row HBM class, SRAM, local bus, directed mesh, or enabled ingress service. They are assumptions unless independently measured. |
| `rho_sram` | `(0,1]` fraction | Limits compile-time live workspace to `rho_sram × SRAM bytes per Core`; it does not change bandwidth. |
| `launch_ns_per_macro`, `memory_setup_ns_per_macro` | Non-negative ns | Add declared startup/setup work. A zero setup value is an uncovered/omitted term, not measured zero. |
| `kappa_quant_eqops`, `kappa_acc_convert_eqops` | Non-negative equivalent operations per item | Price the fixed target quantization/packing and accumulator-conversion templates. They are not universal instruction counts. |
| `performance_profile.explicit_ingress` | Required only for `explicit_ingress_v2` | Supplies source identity, two raw B/s rates, row→ingress mapping, shared-group IDs, enabled flags, and weighted source columns. Shared resources must agree on their declared rate. |

The 64 Core rows are compiled from the source architecture document and checked against the `core64_v2` digest. The resolver derives `N_core=Nx×Ny`, `L0`, INT4 and Vec peaks, SRAM capacity, Die geometry, two HBM rows, bump/D2D budgets, module fit, and system-region totals. For the current source formulas, one HBM stack contributes 36×10⁹ bytes and 1.1776×10¹² B/s; `n` stacks per row means `2n` stacks total. The source Vec rate is 3.9554×10⁹ FP32-equivalent operations/s per lane. These are source inputs, not measured target rates.

The runtime exposes thirteen shared service domains in this stable order:

| Index | Resource | Current source-rate construction |
|---:|---|---|
| 0 | `COMPUTE_GATE` | One second of compute service per second; compute work is expressed in service-seconds. |
| 1–2 | `SRAM_READ`, `SRAM_WRITE` | `N_core × 128e9 × eta_sram` B/s for the fixed read/write port assumption. |
| 3 | `LOCAL` | `N_core × (NoC_bits/8) × 1e9 × eta_local` B/s. |
| 4–5 | `HBM0`, `HBM1` | One per row, each `n × 1.1776e12` B/s before task-class efficiency work. |
| 6–7 | `INJECT0`, `INJECT1` | Disabled for `distributed_ideal_v2`; topology-derived or explicitly supplied and efficiency-scaled for bounded ingress. |
| 8–11 | `XPOS`, `XNEG`, `YPOS`, `YNEG` | Directed row/column cross-section times `(NoC_bits/8) × 1e9 × eta_mesh`; absent axes are disabled. |
| 12 | `HOP` | Directed mesh edge count times per-link service rate. |

Topology profiles are sensitivity scenarios. In `distributed_ideal_v2`, ingress resources are disabled and therefore do not constrain HBM service. `edge_single_word_v2` derives a side-rate from `Nx`, link width, and `eta_inject`. `explicit_ingress_v2` scales supplied ingress rates and validates shared groups, row mapping, and source weights. All current profiles retain `physical_topology_validation: unverified` until supported by physical evidence.

## Task compilation and standalone estimate

The public compiler accepts normalized `coarse_die_task_v2` JSON and the supported legacy `DieOpSpec` v1 envelope. The normalizer preserves shapes, dtype roles, span/query positions, TP slice, layer/weight counts, fusion policy, residency/readiness identity, graph identity, causal/math metadata, and the original task in `source_task`. Unknown fields and unsupported task forms are rejected instead of being silently priced as a generic region. `region_ops` without a typed graph is unsupported.

`configs/example_tasks.json` is an intended offline smoke batch, not an expected-latency fixture; it has not been executed against a linked library from the current source snapshot. It includes a 4-token GEMM (`M=4,N=128,K=128`) with INT4 weights, BF16 activations/output, and FP32 accumulation, plus v2-compiler compatibility envelopes for RMSNorm, QKV+RoPE, attention, residual, SwiGLU FFN, and logits. The transformer examples use hidden size 128 (`4` query heads × head dimension `32`), two KV heads, FFN width 256, vocabulary 512, four query tokens, and sixteen KV tokens. `qkv_rope` is the combined QKV projection/RoPE operator in that grammar; the sample does not claim a standalone RoPE kernel. Those compatibility envelopes target the v2 library and do not call the legacy `extern/die_model` runtime.

`compile` enumerates a bounded deterministic set of mapping candidates, builds each candidate as an immutable `Plan`, runs each through the same standalone runtime used by `estimate_alone`, then selects by:

```text
(ceil(estimated_latency_ns), hbm_payload_bytes, hbm_workspace_peak_bytes,
 sram_peak_bytes_per_core, template_id, tm, tn, tk, th)
```

The estimate used for selection is an assumed model estimate. Plans retain a canonical SHA-256 fingerprint and the hardware fingerprint; submission to a runtime with a different hardware fingerprint fails. The selected plan reports candidate count, selected template, selection rule, estimate provenance, and `granularity_validation: not_run` until an independent granularity oracle has actually been run.

The intended arithmetic and buffer accounting follow design §6: task dtypes determine stored byte widths; real tail dimensions determine operation and byte counts; integer operations/bytes use checked wide intermediates; each live allocation contributes to the per-Core SRAM peak; topology/routing profiles, not a hidden scalar penalty, account for declared on-die routes. GEMM mappings, fused/materialized regions, attention spans, INT4 odd tails, and any special cases must be documented against the final compiler source and their tests. A source-code feature or exact output count alone does not prove that the selected granularity matches a discrete implementation.

## Runtime, activities, and endpoints

A `Plan` is immutable and contains at most 64 named epochs. An epoch has two credit buffers and at most three concurrent activities. Each activity contains:

* thirteen non-negative resource-work totals;
* physical HBM payload bytes indexed by `[row][read/write]`;
* integer INT4 operation and Vec-equivalent-operation totals;
* positive `canonical_q`, optional private-time bound, and signed buffer-credit deltas.

The runtime advances only active epochs. It allocates progress by a weighted max-min active-set calculation, normalized by `canonical_q` within each owner and constrained by the thirteen rates, private-time bounds, and credit-boundary directions. A saturated resource freezes the affected flows; when activity work or a credit boundary is reached, the runtime settles epoch transitions and reallocates. The implementation has a fixed 256 active-set iteration ceiling and returns `numerical_failure` on solver or event-budget failures rather than returning a partial estimate.

Public time is integer nanoseconds, with a sub-nanosecond internal event position. `advance_to` is monotone, `next_event` reports the next public tick, capacity changes first advance under the old capacity, and cancellation first advances to its timestamp. HBM served bytes and equivalent operations accumulate by progress and are returned both as numeric values and exact decimal strings. Runtime counters report activity/owner counts, HBM/compute busy durations, row sharing, served bytes, current capacities, and endpoint/transaction registry sizes. Capacity updates cannot enable a hardware-disabled domain.

An endpoint JSON requires `endpoint_id`, `transaction_id`, `owner_kind` (`comm`, `restore`, `pool`, or `die`), `direction`, `physical_bytes`, and `path_class`. The current paths are `hbm_to_core`, `core_to_hbm`, `hbm_to_edge`, `edge_to_hbm`, `remote_pool_only`, and `noc_only`. Core paths require explicit thirteen-domain route work; HBM edge paths require an explicit `internal_route` boolean; `noc_only` requires explicit route work; `remote_pool_only` has no local route. HBM paths require `eta_hbm` in `(0,1]`. `row_fraction` defaults to 0.5/0.5; row 0 receives `floor(total_bytes × fraction0)` and row 1 gets the remainder. This is a declared allocation, not a physical address map. Duplicate live endpoint IDs and mixed owners within one transaction are rejected.

## C++ and C ABI contract

The C++ entry points are in `include/coarse_die/{Hardware,Plan,Execution}.hh`. The public C declarations are in [Capi.h](../include/coarse_die/Capi.h), with implementations in `src/Capi.cc`. The ABI is version 2. Opaque `DieHandle` values are monotonically allocated IDs rather than object addresses. JSON result buffers returned by the `const char*` result functions are UTF-8, allocated by the library, and must be released with `die_coarse_free_result_v2`. On a null result or nonzero status, `die_coarse_last_error_v2` returns the calling thread's last structured error JSON through a borrowed pointer; do not pass that pointer to the free function.

The handle flow is:

1. `die_coarse_create_model_v2` parses and validates the full hardware descriptor; `hardware_summary` returns input, derived hardware, and fingerprint.
2. `compile_batch` takes `{"specs":[task,...]}` and returns positional `results`/`errors`. Successful results include a retained `plan_handle`, `plan_hash`, and summary; release each plan handle. Each successful occurrence receives a fresh handle, including repeated cache hits. The cache retains escaped hash and summary JSON fragments so a hit can assemble the response without copying and serializing the summary tree again; positional errors, canonical field order, integer Plan v2 and fractional Plan v3 markers are preserved. `estimate_batch` returns positional standalone estimates without exposing retained plan handles; its compiled plans may still enter the model's LRU cache. The model has a bounded LRU compile cache (4,096 entries / 64 MiB conservative retained-allocation accounting). This includes Plan/container capacities, retained response-string capacities and object/allocator allowances; it is not host RSS. Plan handles and temporary batch buffers have separate lifetimes, so an entry evicted later in the same batch does not invalidate its earlier response or returned Plan handle.
3. `create_runtime`, then `submit_plan` or `submit_endpoint`, bind execution handles to one runtime. `advance_to`, `next_event`, `cancel_execution`, and `update_capacity` drive the event state. `query_progress` takes an execution handle; `query_counters` takes a runtime handle.
4. `export_snapshot` serializes a versioned runtime snapshot, including exact internal time, public time, capacities, per-execution plan/progress/credit/served counters, endpoint and transaction registries, event-order version, and rounding version. `import_snapshot` validates the hardware fingerprint, the exact/public time relation, plan/activity/credit state, and endpoint-owner/refcount consistency, then rebuilds service allocation. It returns a new runtime. `fork_runtime` makes a private copy. Use `list_executions` after an import/fork to obtain fresh process-local ABI execution handles. `estimate_on_snapshot` forks privately, submits one candidate and an overlay sorted by `(event_tick, priority, producer, sequence, kind, owner)`, and reports `projection_scope: known_local_world`; it does not invent unknown future traffic or clone external network/pool state. Overlay event kinds are `capacity` (`resource`, `rate`), `cancel` (`local_handle` or a live-world `execution_handle`), `endpoint` (`endpoint` JSON and `owner`), and `plan` (`task` and `owner`). Each event supplies `event_tick`; priority/producer/sequence are optional ordering fields.
5. Release completed execution handles, plan handles, drained runtime handles, and finally the model. Destroying a runtime with active execution or a model with live plans/runtimes returns `DIE_BUSY`.

`DieOwnerV2` requires `struct_size`, `schema_version=2`, and a positive generation. Status codes distinguish invalid input/hardware, missing topology, unsupported forms, overflow, schema/fingerprint mismatch, invalid time, busy/stale handles, numerical failure, and unsupported legacy state. The declarations `DieActivityV2` and `DieCompletionV2` specify layouts but are not currently accepted or returned by a public function; the active ABI path is JSON task/endpoint input and JSON result output.

C ABI registry operations are serialized by a process-wide recursive mutex. This prevents concurrent registry mutation; callers must still impose a deterministic call order for events submitted to the same runtime. Do not race event-sequence mutation and expect a specific winner. `last_error` is thread-local and its pointer remains valid only until the next same-thread ABI operation that changes the error string.

Two capacity queries accept one task JSON and return owned JSON without allocating Plan handles. `die_coarse_workspace_envelope_v2` returns the maximum exact workspace among that task's feasible Mapping candidates. `die_coarse_workspace_capacity_bound_v2` instead uses the same checked-u128 lifetime formulas before SRAM feasibility filtering. Its `capacity_model_revision` is `workspace_lifetimes_v2`, and `capacity_bound_kind` is `all_template_lifetimes_without_sram_filter`; it returns an exact uint64 bound and normalized task/hardware fingerprints. The capacity response schema remains version 1; clients must require the exact model revision and reject v1 responses rather than mixing certificates. The latter is suitable for a proved family of future task shapes, but it does not certify that any mapping is executable. Larger input size alone does not prove coverage: attention rectangles, query counts and request spans must be bounded together. For materialized attention, score/probability/cast workspace is now the peak for one invocation: `layer_instances` and `weight_passes` still scale total physical work and bytes, while their sequential scratch is reusable. Compact span counts and batch remain part of that invocation's live request set. The full-score materialization template and dataflow are version 3 (`full_score_materialization_v3`); the standalone QK template is `attention_qk_materialized_v3` and uses that dataflow ID. Those Plan identities also carry `workspace_model_revision=workspace_lifetimes_v2`. Other attention template/dataflow IDs remain unchanged. v1 Plan/capacity certificates must not be reused with the revised identities. The query does not mutate the compile cache.

The LRU finite-manifest adapter installs one immutable bound per rank before KV admission. Scalar fit uses `base + max(active_workspace, prospective_workspace) <= capacity`; the future bound is not an active lease or KV allocation. The actual compiled Plan must fit that bound before acquisition. Source transport completion gates consumers that reuse released capacity. Coverage and observed limits are documented in [the reproduction package](analysis/lru-future-workspace-20261014/README.md).

## Engineering priors, fit boundary, and TX8 reference

The target prior is in [engineering_prior.json](../configs/engineering_prior.json). Its values follow design §9.2 and remain `assumed`: `eta_pe=0.90`, `eta_vec=0.90`, `eta_hbm_matrix=eta_hbm_vector=0.70`, `eta_sram=1.00`, `eta_local=eta_mesh=eta_inject=0.70`, `rho_sram=0.90`, `launch_ns_per_macro=1000`, `memory_setup_ns_per_macro=0`, `kappa_quant_eqops=4`, and `kappa_acc_convert_eqops=2`. Topology and routing are explicit structural choices; granularity is `not_run`, and physical topology is `unverified`. A zero memory setup value means that setup latency is omitted, not that it was measured at zero.

The current XLSX does not identify these target INT4/FP32 parameters. The reference fit exposes only two conditional quantities for its own profile: one common BF16/FP32 compute efficiency under the FFN timing interpretation, and a vector-memory efficiency under the RMSNorm anchor interpretation. The TX8 values in `calibration_tx8.json` are not copied into target `eta_pe`, `eta_vec`, or HBM efficiencies. The sheet's 13 hard-coded timing groups are not 13 measured datapoints; B16 is the formula over cached group values, and the 19 math nodes do not create 19 independent samples. No independent target samples, warmup/repeat protocol, or measurement-boundary metadata are currently available.

Fit parameters only after adding independent target probes with shape, precision, layout, warmup/repeats, and timing boundaries. Keep structural topology/template choices fixed during a local fit. Before freeing a parameter, demonstrate that the relevant sensitivity columns are identifiable on the calibration set and retain independent held-out probes. Reproducing the reference anchor and its dependent formula values is not held-out validation.

## Five-repository integration and commands

The synchronization utility publishes the same `die/CMakeLists.txt`, `include/`, `src/`, and `configs/` files under each repository's `extern/die_model/coarse/` and writes a content manifest. Run it from the workspace root:

```bash
python3 die/tools/sync_repositories.py
python3 die/tools/sync_repositories.py --check
```

The five repositories are `astra-sim-wscllm`, `astra-sim-wscllm-LRU`, `astra-sim-face`, `astra-sim-face-LRU`, and `astra-sim-joint`. Existing `extern/die_model` is a legacy compatibility library; a root `add_subdirectory(extern/die_model)` builds v2 only if that directory's CMake file includes the synchronized `coarse/` subdirectory. The worktree's current integration state must be read from release manifests, CMake files, and adapter source rather than inferred from the synchronization script.

For each repository use its actual configured entry point, then run the relevant CTest and Python suites. The active new JOINT tree uses the tracked aggregate scaffold, rather than assuming the old JOINT build layout. Its focused native/domain/window checks and real-library Python facade checks are reproducible with:

```bash
bash die/docs/analysis/native-world-snapshot-20261014/run_checks.sh
```

The complete Online build and tiny workload runner are in [the Online reproduction package](analysis/native-local-projection-20261014/online-reproduction/README.md); the [native snapshot package](analysis/native-world-snapshot-20261014/README.md) describes its same-frozen-input session scope. Production Sys/ledger/network/pool live import is not established by these focused checks. Use each other active repository's own documented entry point and suite. The required evidence is per repository: same shared-source digest, build with the v2 C ABI, C++/C ABI plan identity, actual workload behavior, one-owner endpoint accounting, cancellation/drain/release, and snapshot/import/fork replay. A green legacy suite alone does not meet this gate.

## Evidence ledger

These statuses are evidence labels, not completion claims. Update them only from logs created by the source revision being described.

| Gate | Current status | What the status supports |
|---|---|---|
| XLSX cached cells and formula reconciliation | `passed` for `./die/tools/calibrate_tx8.py` | Reproduces workbook cache/source accounting, with `<1e-6 μs` reference-formula closure. No device-accuracy conclusion. |
| TX8 reconstruction (13 groups, Attention/FFN/block) | `passed` as a source reconstruction | Stable AV/PV source mapping and no duplicate block total. Anchor/conditional identities remain explicit. |
| Compiler/Mapping/Plan syntax checks | `passed` as individual translation-unit checks | This is source syntax evidence only; it does not establish linkability, operator-count correctness, or runtime semantics. |
| Public Release build and C ABI compatibility | `passed`: 30/30 CTest; 62/62 Plan hash/summary replay; old envelope 224 responses and 16 errors unchanged | Linked public source and the specified compatibility fixtures are verified. |
| Runtime and LRU admission integration | Public/focused tests and two actual LRU Online fixtures pass | 480/560 typed Plans fit the immutable manifest bounds; remote-store pressure and every business path are not covered by these Online runs. |
| v2 granularity/discrete oracle | `failed` for production E5; E8/E16 unavailable | Original 3%/5% limits remain unchanged. Research prototypes do not replace this gate. |
| Five v2 repository integrations | `in_progress`; release sync, five builds, full CTest, and COMP+COMM/RESTORE metric fixtures are not complete | Existing CMake and adapter tests that exercise legacy `DieModel` do not establish v2 integration. |
| Held-out target hardware accuracy | `not_available` | No eligible independent target measurements are in this workspace. |
| Legacy joint baseline | `passed` pre-change only: CTest 28/28; pytest 72 passed and 12 subtests passed | The log used previously built old binaries, so it is a regression baseline only. |

Engineering-complete requires the design §15 software gates 1–8 to pass on the final source snapshot. The current checkpoint has not passed the complete five-repository business and timing gates. Even after those pass, absent target measurements, the result remains an assumed architecture experiment: it does not establish calibrated target performance or physical topology/order accuracy.
