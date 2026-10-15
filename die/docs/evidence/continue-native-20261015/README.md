# JOINT dependency recovery and isolated preview — 2026-10-15

The real protobuf environment blocker is resolved. The isolated SDK preview built both ABIs, native projection, four native/window tests and the Online binary; native/adapter suites and the tiny legacy Online run passed. The coarse Online probe still hit the original provider guard and failed closed.

**This preview is `before-credit-endpoint-guard`.** It binds JOINT source commit `5c2de289bc5744e5c252d782901cdba8b6f62459` and the 30-file SDK digest **`56e684c09eca0d6d3aa2677f03775219d05611c536b6546de741bd8f24e0f948`**. The main agent subsequently changed credit-endpoint checks and analytical routes; those later public-source changes are **not certified by this preview**. No production source/vendor tree was changed or published here. Current authoritative preview results are in `elementwise-preview-status.json`.

## Actually run

| Check | Actual result |
|---|---|
| Fresh Release aggregate configure | passed with existing real protobuf paths |
| Native/projection/both ABIs build, `-j1` | passed, 135.37 s wall time |
| Four native/window CTest | **4/4 passed**, 0 skip |
| Native Python suite | **6/6 passed**, 0 skip; 3 wire/validation and 3 actual native/CABI integration cases |
| Config resolver suite, explicit real ABI paths | **13/13 passed**, 0 skip |
| Adapter/workspace suite, explicit real ABI paths | **45/45 passed**, 0 skip; includes existing builder units/mocks as designed |
| Online and joint_ledger_cabi build, `-j1` | passed, 444.83 s wall time |
| Tiny legacy Online | C++ **0**, Python **0**; accepted/completed **1/1**, **192/192** successful terminals, 0 skipped/other |
| Tiny coarse_intended Online | C++ **134**, Python **1**, wrapper **1**; original provider guard preserved |

The coarse rejection states that the current JOINT provider uses legacy `hbm_share_divisor/service_factor` and the frozen native projection has not been validated against the coarse shared Runtime. This is an observed blocked integration, not a completed coarse Online run. No sh30/liveworld guard was removed or bypassed. Raw C++/Python logs for both profiles are retained in `elementwise-online-legacy/` and `elementwise-online-coarse_intended/`, including the real abort/rejection. Build/phase commands, exits and durations are preserved individually. Durations above are build diagnostics, not E2E CPU budget results.

`elementwise-stage-identity.json` records the archived JOINT source identity and SDK manifest. `elementwise-final-artifacts.json` binds all 12 built outputs, ELF/build IDs, dynamic dependencies and actual CMake link/flags files. `elementwise-generated-inputs.json` and `elementwise-tiny-inputs.json` retain schema/input identities. The tiny helper's `source_head` is its explicit repository-root metadata; the actual SDK identity is always the separate frozen digest above.

No later-source integration, production coarse Online beyond the guard, liveworld/sh30 validation, official five-repository publish, E2E CPU budget or hardware calibration was run here.

A pytest invocation initially created an incidental root `.pytest_cache`. All four files were proven to belong exclusively to these 45 staged tests and atomically moved intact into this task's work directory; subsequent invocations disable cacheprovider. No other task data was removed. See `elementwise-pytest-cache-note.json`.

## Dependency evidence

The previous Online failure was a stale configured path: `explicit-input-integration-20261015/native-incomplete/build.log` ends with no rule to make the now-absent `/workspace/template/work/protobuf-root/usr/lib/x86_64-linux-gnu/libprotobuf.so`. Current `/workspace/cloud-setup/sysroot/usr` already contains headers, protoc **3.21.12**, libprotobuf.so → libprotobuf.so.32.0.12, and libprotoc.so.32 → libprotoc.so.32.0.12. No network install or system modification was needed. Use a new build directory and real paths; do not recreate the disappeared prefix or reuse its CMake cache.

The independent small dependency probe passed all **9 commands**: version/ldd checks, CMake discovery, generation/compilation/linking of the current JOINT schema, C++ Node serialization roundtrip, Python generation and roundtrip on Python protobuf **6.33.5**. Identities/logs are in `dependency-check.json` and `dependency-*.log`. The Debug probe's optional `IMPORTED_LOCATION_RELEASE` query is absent; the resolved library and actually linked ELF point to the existing library and passed. JOINT uses the same module `${Protobuf_LIBRARIES}` link contract.

`activate.sh` supplies PATH, LD_LIBRARY_PATH and CMAKE_PREFIX_PATH. For pkg-config separately, both `PKG_CONFIG_PATH=/workspace/cloud-setup/sysroot/usr/lib/x86_64-linux-gnu/pkgconfig` and `PKG_CONFIG_SYSROOT_DIR=/workspace/cloud-setup/sysroot` are needed because its `.pc` has `prefix=/usr`. The actual preview instead passes:

```text
-DProtobuf_INCLUDE_DIR=/workspace/cloud-setup/sysroot/usr/include
-DProtobuf_LIBRARY=/workspace/cloud-setup/sysroot/usr/lib/x86_64-linux-gnu/libprotobuf.so
-DProtobuf_PROTOC_EXECUTABLE=/workspace/cloud-setup/sysroot/usr/bin/protoc
-DProtobuf_PROTOC_LIBRARY=/workspace/cloud-setup/sysroot/usr/lib/x86_64-linux-gnu/libprotoc.so.32
```

`PROTOBUF_FROM_SOURCE` was unset to use the repository's existing FindProtobuf module branch. The historical frozen89 binaries/results retain their original scope and are not reused as tests of this SDK.

## Reproduction

`elementwise-sdk-from-5c2de28.patch` preserves the exact historical preview SDK as a small reviewable diff from the retained base commit. It was actually reapplied to a fresh archived base under work, then **all 30 files and the canonical digest above were reverified**; see `elementwise-sdk-patch-identity.json`. This keeps the tested historical source reconstructible after ignored work folders disappear, even though the current public source includes later fixes. The patch is evidence and must not replace current production source.

For a new isolated preview, first supply an already frozen/validated SDK directory and exact committed JOINT source:

```bash
source /workspace/cloud-setup/activate.sh
python3 die/docs/evidence/continue-native-20261015/prepare_native_stage.py \
  --root . --sdk /absolute/path/to/frozen/die-sdk \
  --source-commit EXACT_COMMIT_SHA --tag NEW_UNUSED_TAG
bash work/native-dependency-check/NEW_UNUSED_TAG/run_phase.sh configure
bash work/native-dependency-check/NEW_UNUSED_TAG/run_phase.sh build
bash work/native-dependency-check/NEW_UNUSED_TAG/run_phase.sh native_ctest
bash work/native-dependency-check/NEW_UNUSED_TAG/run_phase.sh native_python
bash work/native-dependency-check/NEW_UNUSED_TAG/run_phase.sh online_legacy
bash work/native-dependency-check/NEW_UNUSED_TAG/run_phase.sh online_coarse_intended
```

The helper archives JOINT and copies SDK files into the new preview only, creates the canonical release manifest, refuses to overwrite a stage, and never publishes/builds automatically. Record fresh logs/ELF identities when executing. `run_phase.sh build` includes Online and uses `-j2`; the actual original execution here split native/Online builds at `-j1`, as recorded by `elementwise-run_steps.py` and each phase metadata. That original runner is specific to this fixed old digest. The separate config/45-adapter commands and exact environment are in their phase metadata; they require Python protobuf bindings, which the recorded `python-schema` phase generated only in the isolated stage.
