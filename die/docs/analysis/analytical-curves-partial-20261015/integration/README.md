# Five-repository ABI and FACE cost-query integration

This is a work-only execution package. `run_integration.py` defaults to a
printed dry-run plan. `--execute` consumes existing build products and runs
the checks; it never builds, publishes, runs the canonical publisher, or edits
repository/vendor source. All generated output belongs in a fresh directory
under `work/`.

## What the run checks

The five-ABI configuration entry point is
`die/docs/analysis/runtime-credit-20261015/validate_five_abis.py`. It requires
exactly the active `astra-sim-wscllm`, `astra-sim-wscllm-LRU`,
`astra-sim-face`, `astra-sim-face-LRU`, and `astra-sim-joint` checkouts; the
map must not contain `astra-sim-joint_old`. For every checkout it passes the
actual `libdie_model_cabi.so` and `libcoarse_die_cabi.so` paths into the
existing `test_config_resolver*.py` suite. The frozen expected counts are
20+15+18+17+13=83 with no skips. The wrapper verifies both library hashes are
recorded for every repository, all five config suites returned the expected
counts, and no suite skipped a test.

The additive coarse query is exported as
`die_coarse_estimate_cost_batch_v2(DieHandle, const char*)`. The two FACE
checkouts have the same Python dispatch in
`sh_test_mesh/workload/llama2_7b_inference/online/die_model_oracle.py` and
`face_scheduler.py`: when that symbol is present the scheduler calls the
cost-only query; when it is absent, `estimate_cost()` calls the established
full-report `die_coarse_estimate_batch_v2` route. `face_cost_abi_probe.py`
loads each actual shared library via `ctypes`, compiles a real example task
through it, checks the cost/hash/status against the full report, and confirms
which endpoint the real FACE scheduler selected. The `old` probe requires an
actual coarse-v2 library that exports the established coarse symbols but
does not export the new cost-only symbol; it does not mock or hide symbols.

The runner also uses the existing exact-source FACE staging entry point,
`die/docs/analysis/face-workspace-20261015/run_python_checks.py`, with both
real ABI libraries and real `protoc` output. It selects the latest recorded
runtime-credit module sets (132 tests for FACE and 151 for FACE-LRU) and
requires zero skips. It records source and binary identities under the new
work result directory.

The baseline old-symbol library in this workspace was built separately from
the untouched pre-publisher `astra-sim-face/extern/die_model` tree, at
`work/analytical-curves/integration/build-baseline-face/lib/libcoarse_die_cabi.so`.
Its source manifest and tree digest are stored beside this document. This
isolated baseline build is useful only for the fallback path; it is not a new
vendor release and says nothing about the pending canonical source.
The direct reproduction commands are:

```bash
source /workspace/cloud-setup/activate.sh
cmake -S astra-sim-face/extern/die_model \
  -B work/analytical-curves/integration/build-baseline-face \
  -DCMAKE_BUILD_TYPE=Release -DCOARSE_DIE_BUILD_TESTS=OFF
cmake --build work/analytical-curves/integration/build-baseline-face \
  --parallel 2 --target die_model_cabi coarse_die_cabi
```

The recorded baseline coarse library SHA256 is
`bf56b1c1b7afd3c91a8cace974db19249d3e82567c3c770eda0e623e741cfddc`; the
legacy library SHA256 is
`7b961c60505ce8e37be3520fe3f363d1cadbe7fb8cf99cced50daa6285690d68`. Both
FACE checkouts have already exercised the fallback against that real coarse
library: the new symbol is absent, both old-symbol queries report the same
361 ns as the full estimate, and the scheduler instrumentation records
`[false]` for its `cost_only` dispatch flag. The raw probe JSON and logs are
retained next to the build identity.

## Running after current builds are ready

First create `build-map.json` with the five actual build directories. Paths may
be absolute or relative to the repository root; each directory must contain
`lib/libdie_model_cabi.so` and `lib/libcoarse_die_cabi.so`. Use the active
JOINT path only. Then source the installed toolchain and review the dry-run:

```bash
source /workspace/cloud-setup/activate.sh
python3 die/docs/analysis/analytical-curves-partial-20261015/integration/run_integration.py \
  --build-map work/analytical-curves/integration/build-map.json \
  --old-coarse-library "$PWD/work/analytical-curves/integration/build-baseline-face/lib/libcoarse_die_cabi.so"
```

When the five current build directories contain libraries from the intended
source identity and the active JOINT build has passed its prerequisite
provider/deadlock checks, execute the exact plan into a fresh work directory:

```bash
source /workspace/cloud-setup/activate.sh
python3 die/docs/analysis/analytical-curves-partial-20261015/integration/run_integration.py \
  --build-map work/analytical-curves/integration/build-map.json \
  --old-coarse-library "$PWD/work/analytical-curves/integration/build-baseline-face/lib/libcoarse_die_cabi.so" \
  --execute
```

The runner creates a timestamped result folder by default. An explicit
`--output-dir "$PWD/work/analytical-curves/integration/run-review-1"` may be
used instead; the directory must not already exist. A direct FACE probe can be
run independently to isolate one new/old ABI branch:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 work/analytical-curves/integration/face_cost_abi_probe.py \
  --repo-root "$PWD/astra-sim-face" \
  --library "$PWD/work/analytical-curves/integration/build-baseline-face/lib/libcoarse_die_cabi.so" \
  --expect old \
  --output "$PWD/work/analytical-curves/integration/face-cost-face-old.json"
```

Replace `--library` with the current FACE build's coarse ABI and `--expect new`
to exercise the new symbol. Repeat for `astra-sim-face-LRU`. The runner
performs all four combinations automatically.

For a native JOINT-focused follow-up, the current active source registers
`JointPipelineWindowTest`, `NativeProjectionCApiTest`,
`NativeCoarseProjectionTest`, and `NativeCoarseDieDomainTest` in the JOINT
aggregate. The corresponding Python entry point is
`online.test_die_model_adapter online.test_coarse_die_adapter
online.test_coarse_die_workspace` from the active JOINT workload directory,
with both ABI environment variables set to that same build. These focused
checks are separate from the five config resolver suites and should only run
after their targets have been built from the intended source.

## Coverage boundary

The integration runner covers actual two-ABI discovery through all 83 config
cases, the two FACE current/legacy query paths, and the recorded focused FACE
Python suites. It does not run an entire production workload. The current
project handoff still lists full production JOINT live-world import and its
`sh30` provider path, E8/E16 and complete GQA coverage, target held-out accuracy
(the current E5 axis remains failed at its frozen threshold), all port/topology
pressure cases, and host-budget certification as open. Tiny/overlap Online
reproductions are available in the existing FACE/JOINT reproduction packages;
their results must retain their own input, source, and binary identities and
must not be inferred from these focused ABI checks.

## Baseline library identity

`baseline-face-vendor-manifest.json` captures all 52 source files under the
FACE vendor tree before the independent build. The isolated Release build uses
the session toolchain (CMake 3.31.10, GCC 14.2.0, protoc 3.21.12); full
configure/build logs and final shared-library hashes are stored in this
directory. No CTest or Python suite was run as part of constructing that
baseline library.
