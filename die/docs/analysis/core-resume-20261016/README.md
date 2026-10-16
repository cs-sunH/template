# Current integration audit (2026-10-16)

This package adds portable reproduction tools and source-identity audit evidence. No five-repository production/vendor files were changed; no SDK was published, no build/performance run was started by this audit, and no archived build product was reused. Tools have syntax/help/dry-run checks only. Defaults write to dedicated repository work/ directories. Historical probe/aggregate bytes and original identities are recorded in `artifact-provenance.json`.

## Evidence that remains valid

- HEAD when inspected: `b8ddbcfd62e03738b0c3996444355cbbe2e99b80`; branch `codex/die-core-resume-20261016`. HANDOFF/RELAY merge conflicts were owned by the parent.
- Seven archived authorized-point identities (FACE build/unit/config/default and FACE-LRU build/config/default) have **zero mismatches**, each over all 30 recorded files. Their ELF SHA was `d0c297e507a266a004e0e430266032e6948ee48a594ee21485f7ee99663516b7`. This preserves FACE 11/11, FACE config21, FACE-LRU config20 and both default-point checks only within those exact scopes.
- All five installed SDKs still match all 24 actual files in their release manifest, digest `23160c512203596611d5c534104f5dd2015be2ef75a2665c88ffe1cfddcc7e9b`. This proves the historical release is unchanged; it does not validate current Core32 SDK functionality.
- Re-executed archived `audit_source.py` against current source: DOCX SHA, 32-row JSON/Core resource constants, compiled legacy table entries and current authorized-point configs match in all five variants. Output: `../../evidence/core-resume-20261016/integration-audit/source-audit-current.json`. This is a source audit, not ABI execution.

## Missing final legacy scope

Current resolver test counts are **21 + 20 + 23 + 18 + 16 = 98** for FACE / FACE-LRU / WSCLLM / WSCLLM-LRU / JOINT. WSCLLM includes the two `test_config_resolver_read_fail_closed.py` tests. The three missing variants require 57 tests, successful default engineering-point checks and fresh legacy ABI identity. Each common C++ legacy suite contains 11 tests; JOINT has four extra internal/K2 tests, total15. FACE-LRU latest unit suite was not run; its per-file C++ legacy sources equal FACE's for the changed hardware implementation, but this is not a separate FACE-LRU unit execution claim.

Parent configured three fresh Make builds, then hit a runner directory collision before build launch. The runner is now fixed to place logs under run/phases/<phase>, keeping run/build separate. Original configure files remain untouched; use a fresh label for final validation. This was a verification-runner failure, not source/ABI failure. See `../../evidence/core-resume-20261016/integration-audit/runner-directory-collision.json`. Parent then independently started fresh full runs with label `core32-final-three-fixed`; their results are outside this source audit. Run:

```sh
source /workspace/cloud-setup/activate.sh
python3 -B die/docs/analysis/core-resume-20261016/run_legacy_validation.py \
  --repo astra-sim-wscllm astra-sim-wscllm-LRU astra-sim-joint \
  --label new-fresh-legacy-validation --scope full --jobs 2
```

A new label with the default phases will refuse any pre-existing build tree and run configure/build/unit/config/default. `--scope abi --phases configure build config default` checks only the legacy ABI/config path; it does **not** certify the new coarse SDK.

## Isolated current SDK preparation

Only run outside the parent's public performance window. These tools have received syntax/`--help`/dry-run checks only. No stage has yet been copied or hashed by them. A dry-run without a prepared snapshot explicitly reports `not_prepared_command_plan_only` and `source_digest=not_frozen_no_snapshot`; it prints paths/commands only. Preparation then freezes the actual current public source digest before execution.

```sh
python3 -B die/docs/analysis/core-resume-20261016/prepare_isolated_sdk.py \
  --output work/core-resume-sdk
python3 -B die/docs/analysis/core-resume-20261016/run_isolated_sdk.py \
  --snapshot work/core-resume-sdk \
  --output work/core-resume-sdk-validation/abi --phases abi
```

The second command prints a dry-run plan. Add `--execute` to configure clean Make builds of both ABIs in every copied variant and run all98 resolver tests, 0skip, plus each actual typed resolver through that variant's current coarse ABI. The preparation manifest freezes current public release digest, all canonical file hashes, source HEAD, original SDK identity and every copied consumer file. `--expected-digest DIGEST` can pin an externally reviewed public digest.

Preparation copies tracked files as actual files, preserving executable modes, instead of symlinking Python source. This matters: the old `stage_five.py` symlink approach lets `Path(__file__).resolve()` escape to original consumers and old installed SDK configs. The new stage copies `sh_test_mesh`, C++ sources and legacy adapters, then replaces only **the new isolated copy** of `extern/die_model/coarse`. Production vendors are never written.

After successful ABI builds, plan and execute focused actual Python consumers:

```sh
python3 -B die/docs/analysis/core-resume-20261016/run_isolated_sdk.py \
  --snapshot work/core-resume-sdk \
  --output work/core-resume-sdk-validation/python --phases python
```

Preparation generates real Chakra Python bindings with `protoc` inside all five copied variants, records each original tracked/copied hash plus generation command/tool/log/output hash, then freezes the complete manifest. WSCLLM has a tracked et_def_pb2.py, so generation must precede freezing. The Python validation phase only checks that frozen binding file/hash, then runs current adapter/scheduler/graph fixtures, real FACE cost-only clients and the actual-response R3/not_available guard probe. It never overwrites a frozen binding. FACE R4 workloads may fail: preserve those failures; do not reduce their layers, hide unavailable body latency or bypass guards. Run the initial dry-run, then add `--execute`.

For actual C++ consumers, plan `--phases native` into a new output directory. The isolated SDK runner uses the frozen generic aggregate with `ASTRA_SIM_ROOT` set to each copied variant. It builds/runs DieModelAdapter, DieExecutionLifecycle, LocalHbmDieMemory and applicable WorkspaceAdmission fixtures for all five; JOINT additionally runs native projection/domain/window, coarse workspace, HBM tail and pipeline mutation, followed by actual native Python projection tests. It does not run full Online or a CPU budget, and never restores joint_old.

Dependencies: C++17/GCC, GNU Make, CMake>=3.22, actual `protoc`/matching C++ Protobuf development package for native aggregate and Python protobuf runtime for generated bindings. System Protobuf uses the default FindProtobuf module; unset `PROTOBUF_FROM_SOURCE` unless a real config package is installed (`True` exactly selects CONFIG mode). Full native builds also use each frozen variant's vendored fmt/spdlog/yaml/json/backend sources. The preparation tool accepts `--protoc`; the validation runner accepts `--cmake`, `--ctest`, `--compiler` and `--jobs`.

## Known harness and acceptance limits

- Historical `run_validation.py` hardcodes old `/workspace/template/work/tool-env` and Ninja paths. Its `--scope abi` builds **only** `die_model_cabi`.
- Historical `analytical-curves-partial-20261015/integration/run_integration.py` has fixed83 expected config tests. It must not be used unchanged for current98; the new work runner uses current counts and zero skips.
- `runtime-credit-20261015/validate_five_abis.py` alone does not reject skipped tests or assert expected counts. It can write success with missing ABI coverage if discovery tests skip; the new runners enforce both.
- Real consumer tests that load `extern/die_model/coarse/configs/example_hardware.json` require a copied current SDK/config tree, not merely setting the new shared-library path while leaving the old descriptor in place.
- Existing standalone SDK builds/ABI, native fixture tests or tiny Online runs do not certify fixed-stream/end-to-end CPU budgets. Budget remains <=10% overhead versus old Roofline for the same workload, with separate fixed-event backend and full policy measurements.
- JOINT coarse Online remains fail-closed by `require_legacy_projection_context`; keep provider/live-world protections. No fixed-stream replay harness or standard sh30 CSV was established by this audit. `explicit-input-integration-20261015/e2e-budget-plan.md` describes those prerequisites.
- No current hardware calibration claim follows from legacy/TX8 scalar compatibility, synthetic oracles or these software integration checks. Legacy default retains4x16GB/410GBps; public current DOCX HBM profile is4x36GB and must retain its identity.

## Saved source evidence

`die/docs/evidence/core-resume-20261016/integration-audit/scope-audit.json` records the seven archived/current source comparisons and the five old SDK manifest/content checks performed earlier in this audit. It references archive members rather than copying old run outputs as new tests. `source-audit-current.json` retains the actual DOCX/table/current-config source audit; `runner-directory-collision.json` records the pre-build tooling failure. None of these files claims new ABI, full business, hardware-accuracy or CPU-budget acceptance.

`integration-audit/sdk-tool-static-checks.json` under the evidence directory records that all selected Python consumer modules exist and all requested native CMake targets are registered in current source. Its status describes only that static audit; native configure/build/link/execute remain not_run by this audit.

## Actual execution updates from the parent

The final three legacy variants passed 37 C++ tests and 57 configuration tests
with zero skips. All five isolated current SDK builds produced both ABIs and
all 98 configuration tests passed without skips. The five coarse shared
libraries are byte-identical to the freshly tested public library. These
results are in `integration-public-checkpoint.json`; formal vendors remain
canonical24 and the new SDK is not published.

The initial generic historical typed probe is unsuitable for JOINT because its
authoring resolver does not expose `derive_coarse_die_hardware`. Its initial
failure is retained as a harness failure. The new
`probe_coarse_hardware_consumer.py` uses JOINT's actual existing online oracle;
it passed the Core32 identity/count checks and all eight retired-selector
rejections. It does not claim that JOINT's authoring resolver now supports a
coarse node. Until the running native matrix has finished, the active runner
file is left unchanged so its recorded script SHA remains honest; the next
runner revision must select this supported entry instead of the old helper.

FACE's initial Python suite found one stale legacy hardware pin (6197 ns).
Seven actual ABI terms gave 6191.5343845817139 ns, with independent Decimal
ceil 6192. The versioned test expectation/comment was corrected. A separate
copied FACE consumer tree differed from the frozen original only in that
one test file; the complete suite then passed 86/86 with the same fresh ABI.
Both original failure and complete correction identity are retained.
FACE-LRU passed80, WSCLLM64 and JOINT92. WSCLLM-LRU passed62 and skipped one
optional test because that repository has no Python oracle client; the strict
zero-skip runner correctly records this as incomplete coverage, not a pass.
Actual C++ consumers are still being built and tested.

The three real FACE/FACE-LRU/JOINT Python oracles correctly reject R3 and the
actual emitted R4 unavailable predictions, including cost/full fallback and
error siblings. No availability guard was removed. Full business correctness,
original busy precision and end-to-end CPU budgets remain open.
