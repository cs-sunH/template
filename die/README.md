# Coarse Die v2

> Development checkpoint: read [docs/HANDOFF.md](docs/HANDOFF.md) before continuing. The default build is blocked by the benchmark target's missing `COARSE_DIE_SOURCE_DIR` definition; operator and five-repository acceptance remains incomplete. This checkpoint is preserved for repair, not a completed release.

`die/` is the canonical source for a coarse architectural model of one Die: a source-derived hardware resolver, a task-to-plan compiler, a shared service runtime, and a C ABI. Results describe a versioned set of engineering assumptions. They are not target-device measurements or a claim of silicon prediction accuracy.

The TX8 reconstruction is a separate reference profile. The workbook has cached values and formulas but no auditable raw timing log; its 246 μs RMSNorm point is a reported anchor, while 3363 μs is a conditional whole-FFN interpretation. The exact reproduction of those source values is accounting verification, not accuracy validation. See [calibration/report.md](calibration/report.md) and [calibration/per_operator.csv](calibration/per_operator.csv).

`configs/engineering_prior.json` records the §9.2 engineering prior for target INT4/FP32 work. The v2 target prior remains separate from the TX8 reference efficiencies. `configs/example_hardware.json` is a complete hardware input; the standalone prior file is metadata and must not be passed as a complete hardware descriptor.

## Build and run

With CMake, a C++17 compiler, and the repository JSON header available:

```bash
source /workspace/cloud-setup/activate.sh
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
| Compiler/Mapping/Plan source syntax checks | `passed` as individual translation-unit checks; this does not establish a linked library |
| v2 Release library build, CTest, and example C ABI estimates | Static/shared library targets compiled during handoff; default all-target build failed at the benchmark macro definition. Full CTest and example estimates remain `not_run`; see committed handoff evidence. |
| Minimal standalone Runtime semantic check | `passed` for the narrow check reported at this checkpoint; full CTest/C ABI snapshot and metric coverage remains `not_run` |
| Five-repository v2 adapter integration | `in_progress`; source wiring is being edited, but release sync, five builds, complete CTest, and COMP+COMM/RESTORE metric fixtures are not yet verified |
| Independent target-hardware held-out validation | `not_available`; no eligible target timing samples are present |
| Legacy joint baseline | `passed` before the current v2 work: 28/28 CTest and 72 pytest tests plus 12 subtests. It ran old binaries and is not v2 verification. |

The engineering project remains incomplete: the full-library, C ABI, and five-repository acceptance gates have not passed. Exact reference reconstruction, a narrow Runtime semantic check, or source-table golden checks cannot upgrade target performance accuracy or physical-topology evidence.
