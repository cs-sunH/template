# Coarse Die v2

> Development checkpoint: read [docs/HANDOFF.md](docs/HANDOFF.md) for the source identity and actual tests. Explicit single-Die hardware inputs are implemented; model accuracy, sustained speed and five-repository workload integration remain unaccepted.

`die/` is the canonical source for a coarse architectural model of one Die: a source-derived hardware resolver, a task-to-plan compiler, a shared service runtime, and a C ABI. Results describe a versioned set of engineering assumptions. They are not target-device measurements or a claim of silicon prediction accuracy.

The hardware authority is [架构建模_单Core版_严格复核修订.docx](../架构建模_单Core版_严格复核修订.docx), source revision `41f119135fd2d4cc87292f59b4b00adc918eece0`. There is one Core design. SRAM capacity and NoC width select one of the 32 rows in `configs/core32.json` (`core32_v1`); the Core edge is always the SRAM long edge, `L0=a`. `architecture.core_type` is removed from the public input: its presence is an error for every value. Prior hardware descriptors, cached Plans and snapshots retain their historical identities and cannot be relabeled as this hardware revision.

The hardware DOCX does not define simulator timing policy. The root simulator design §1.5, §1.6, §8.2.1 and §13 continue to specify analytical workload, bounded cumulative service curves, accuracy and speed requirements. Hardware-document migration alone establishes no new verification result.

The TX8 reconstruction is a separate reference profile. The workbook has cached values and formulas but no auditable raw timing log; its 246 μs RMSNorm point is a reported anchor, while 3363 μs is a conditional whole-FFN interpretation. The exact reproduction of those source values is accounting verification, not accuracy validation. See [calibration/report.md](calibration/report.md) and [calibration/per_operator.csv](calibration/per_operator.csv).

`configs/engineering_prior.json` records the §9.2 engineering prior for target INT4/FP32 work. The v2 target prior remains separate from the TX8 reference efficiencies. Both `configs/example_hardware_explicit.json` and `configs/example_hardware.json` now provide explicit 21×29 mm single-Die inputs without a type field or `search_profile`. The latter also retains an optional 215×215 mm historical system-region diagnostic, labeled `derived.system.layout_profile=legacy_region_fit_10pct_v1` with `layout_provenance=compatibility_assumption_not_hardware_docx`; its 10% space reservation is not a source constant, single-Die prerequisite or hardware-selection rule. The standalone prior file is metadata and must not be passed as a complete hardware descriptor.

The user-authorized compute-reference example is
[`configs/tx8_compute_matched_hardware.json`](configs/tx8_compute_matched_hardware.json):
0.75 MiB, 512-bit NoC, 10×10 Core, and a 25×32 mm Die (800 mm²). Its
126.6 INT4 TOPS nominal matrix peak differs numerically by 1.09375% from the
TX8 128 BF16 TFLOPS reference; the operation types differ, so this is an
engineering comparison point, not equivalent measured throughput. Its Vec
peak is separately 12.65728 trillion FP32-equivalent operations/s; Vec and
memory are not matched to TX8. The public example explicitly uses four DOCX
HBM3E stacks, while the five legacy examples retain their stated four-stack
16 GB/410 GB/s compatibility profile. Neither is a target hardware calibration.
The choice was made once under user authorization; the simulator still does not
select hardware. See the [design record](docs/evidence/single-core-20261016/tx8-compute-design.json).

## Hardware is supplied by the user

Provide Die width/height and the Core grid (`core_columns_nx`, `core_rows_ny`),
along with SRAM, NoC width, HBM and the explicit performance profile.
The resolver derives area, capacities and the 13 resource rates for that one
instance. It does not enumerate hardware, choose an area/grid, or recommend a
Pareto winner. Wafer geometry and `search_profile` are not required. Optional
`validation_constraints` can reject the supplied instance without changing it.
Software tile/residency choices within that fixed hardware remain supported.

The current prediction-error target is 10%. After multiple recorded tests and
reasonable calibration attempts, results within 20% may be accepted with their
scope and unmet 10% target disclosed. This replaces the historical 3%/5% gates;
exact work/capacity accounting, the busy definition and speed budgets remain.
Calibration uses the workbook's correctly decomposed operator data and a small
number of identifiable parameters; reference data and target priors stay distinct.

## Build and run

With CMake, a C++17 compiler, and the repository JSON header available:

```bash
cmake -S die -B build/coarse_die_v2 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCOARSE_DIE_JSON_INCLUDE="$PWD/astra-sim-wscllm/extern/helper"
cmake --build build/coarse_die_v2 -j
ctest --test-dir build/coarse_die_v2 --output-on-failure
./die/tools/estimate.py \
  --library build/coarse_die_v2/lib/libcoarse_die_cabi.so \
  --hardware die/configs/example_hardware_explicit.json \
  --tasks die/configs/example_tasks.json \
  --output -
```

The C ABI shared object is written to `build/coarse_die_v2/lib/libcoarse_die_cabi.so`. `estimate.py` calls the public `create_model_v2` and `estimate_batch_v2` APIs, prints one status-bearing item per task to stdout, and returns nonzero if any item fails. The same standard-library `ctypes` entry point accepts a single task object or a `{"specs":[...]}` task file; `--output PATH` writes JSON to a file. For the detailed config/task schema and lifecycle contract, see [docs/implementation.md](docs/implementation.md).

A top-level estimate status of `not_available` means its numeric body timing is
diagnostic and cannot be used as a complete latency prediction. The CLI retains
that body, reports an item error and exits nonzero; `status_code: 0` still denotes
a successful underlying ABI call. Nested target-calibration/heldout availability
does not determine whether the model prediction is available. Repeated regions
with unmodeled nonservice windows currently use this status.

C++ production consumers call `coarse_die::require_complete_prediction(*plan)`
from `coarse_die/Prediction.hh` before acquiring resources or submitting a Plan.
The public compiler and Runtime retain body execution for diagnostics. The five
adapters enforce availability at their compilation/admission/submission and
supported restore boundaries; their old SDK compatibility is tested separately
from isolated new-SDK integration.

To rebuild the XLSX XML audit independently of the C++ library:

```bash
./die/tools/calibrate_tx8.py
```

It regenerates `configs/calibration_tx8.json`, `calibration/tx8_samples.json`, `calibration/report.md`, and `calibration/per_operator.csv`. It reads cached formula results and formula metadata; it does not estimate hardware performance.

## Integrating the canonical source into the five AstraSim trees

The canonical-source synchronization utility is:

```bash
python3 die/tools/sync_repositories.py
python3 die/tools/sync_repositories.py --check
```

The first command vendors the same `die/CMakeLists.txt`, `include/`, `src/`, and `configs/` snapshot into each repository's `extern/die_model/coarse/`; the second checks file and release-manifest hashes. It leaves each existing `extern/die_model` compatibility implementation in place. Formal vendors are currently pinned to canonical24; new-source checks use isolated preview trees until the model and integration gates pass. Configure and build each repository with its own existing CMake entry point; the joint example is:

```bash
cmake -S astra-sim-joint/build/astra_analytical \
  -B astra-sim-joint/build/astra_analytical/build_congestion_aware \
  -DBUILDTARGET=congestion_aware \
  -DNETWORK_BACKEND_BUILD_AS_LIBRARY=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build astra-sim-joint/build/astra_analytical/build_congestion_aware -j
ctest --test-dir astra-sim-joint/build/astra_analytical/build_congestion_aware \
  --output-on-failure
```

Apply the same configure/build/test sequence to `astra-sim-wscllm`, `astra-sim-wscllm-LRU`, `astra-sim-face`, and `astra-sim-face-LRU`, using each repository's own `build/astra_analytical` tree and supported `BUILDTARGET`. Vendoring is not proof that an adapter uses v2 at runtime; each repository needs its own C ABI/adapter, workload, endpoint ownership, and snapshot/replay evidence. Current integration status is recorded in the implementation guide.

## Evidence status

| Evidence | Status |
|---|---|
| Workbook cached source/formula reconciliation | `passed` for the independent Python audit; source values are not raw measurement evidence |
| TX8 13-group/Attention/FFN reconstruction | `passed` as conditional source-formula reproduction; not performance accuracy |
| Public Release build, CTest and Plan compatibility | Current availability/CLI and shared-model checks are recorded in HANDOFF; historical 30/30 and 62/62 exact Plan comparisons apply to their capacity-bound source checkpoint |
| Capacity-only compiler bound and LRU admission | `passed` for the finite request-aggregated manifest fixtures; both actual Online runs finished, with all 480/560 typed Plans below their immutable rank bounds |
| Runtime, endpoints and snapshot semantics | Focused public and adapter tests pass; these do not establish production timing accuracy or every business path |
| Five-repository v2 adapter integration | `in_progress`; canonical publication and per-variant ABI/config checks are recorded separately from workload/runtime integration |
| Production timing granularity | Synthetic single-Core comparisons have been reevaluated under 10%/20%; long R1 completion improved, while busy and other covered cases still fail. These comparisons are not hardware calibration |
| Independent target-hardware held-out validation | `not_available`; no eligible target timing samples are present |
| New active joint | Additive coarse adapter/runtime work uses `astra-sim-joint`; `astra-sim-joint_old` was removed and must not be restored. Current verification scope is in HANDOFF. |

The engineering project remains incomplete. Follow the design §1.4 formula-first principle: auxiliary details that are not directly tied to hardware design or software task inputs use simple mathematical formulas where sufficient. Necessary parameter coverage, readiness and shared-resource constraints remain required. Current source-bound evidence and reproduction entries are in [docs/HANDOFF.md](docs/HANDOFF.md), including [consumer availability](docs/analysis/consumer-availability-20261016/README.md) and the retained model/performance failures.

The evidence above retains its original source scope. Historical fixtures and published five-repository SDKs do not become validation of `core32_v1` through a documentation or identity change; the migration needs its own source-bound checks before publication. Existing legacy geometry decoding is compatibility behavior, not the input rule for a new hardware instance.
