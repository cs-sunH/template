# Coarse Die v2

> Development checkpoint: read [docs/HANDOFF.md](docs/HANDOFF.md) before continuing. The default public build and 30 CTest tests pass. Production timing accuracy and complete five-repository business integration remain unaccepted.

`die/` is the canonical source for a coarse architectural model of one Die: a source-derived hardware resolver, a task-to-plan compiler, a shared service runtime, and a C ABI. Results describe a versioned set of engineering assumptions. They are not target-device measurements or a claim of silicon prediction accuracy.

The TX8 reconstruction is a separate reference profile. The workbook has cached values and formulas but no auditable raw timing log; its 246 μs RMSNorm point is a reported anchor, while 3363 μs is a conditional whole-FFN interpretation. The exact reproduction of those source values is accounting verification, not accuracy validation. See [calibration/report.md](calibration/report.md) and [calibration/per_operator.csv](calibration/per_operator.csv).

`configs/engineering_prior.json` records the §9.2 engineering prior for target INT4/FP32 work. The v2 target prior remains separate from the TX8 reference efficiencies. `configs/example_hardware.json` is a complete hardware input; the standalone prior file is metadata and must not be passed as a complete hardware descriptor.

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
  --hardware die/configs/example_hardware.json \
  --tasks die/configs/example_tasks.json \
  --output -
```

The C ABI shared object is written to `build/coarse_die_v2/lib/libcoarse_die_cabi.so`. `estimate.py` calls the public `create_model_v2` and `estimate_batch_v2` APIs, prints one status-bearing item per task to stdout, and returns nonzero if any item fails. The same standard-library `ctypes` entry point accepts a single task object or a `{"specs":[...]}` task file; `--output PATH` writes JSON to a file. For the detailed config/task schema and lifecycle contract, see [docs/implementation.md](docs/implementation.md).

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

The first command vendors the same `die/CMakeLists.txt`, `include/`, `src/`, and `configs/` snapshot into each repository's `extern/die_model/coarse/`; the second checks file and release-manifest hashes. It leaves each existing `extern/die_model` compatibility implementation in place. Configure and build each repository with its own existing CMake entry point; the joint example is:

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
| Public Release build, CTest and Plan compatibility | `passed`: 30/30 tests; 62/62 prior Plan hashes and complete summaries unchanged at the capacity-bound checkpoint |
| Capacity-only compiler bound and LRU admission | `passed` for the finite request-aggregated manifest fixtures; both actual Online runs finished, with all 480/560 typed Plans below their immutable rank bounds |
| Runtime, endpoints and snapshot semantics | Focused public and adapter tests pass; these do not establish production timing accuracy or every business path |
| Five-repository v2 adapter integration | `in_progress`; canonical publication and per-variant ABI/config checks are recorded separately from workload/runtime integration |
| Production timing granularity | `failed`: E5 finish/busy oracle remains outside the original 3%/5% limits; E8/E16 evidence is unavailable |
| Independent target-hardware held-out validation | `not_available`; no eligible target timing samples are present |
| New active joint | Additive coarse adapter/runtime work uses `astra-sim-joint`; `astra-sim-joint_old` is archival. Current verification scope is in HANDOFF. |

The engineering project remains incomplete. Follow the design §1.4 formula-first principle: auxiliary details that are not directly tied to hardware design or software task inputs use simple mathematical formulas where sufficient. Necessary parameter coverage, readiness and shared-resource constraints remain required. Current evidence and reproduction instructions are in [docs/evidence/lru-future-workspace-20261014](docs/evidence/lru-future-workspace-20261014/README.md).
