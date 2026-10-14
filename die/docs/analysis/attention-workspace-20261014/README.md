# Attention workspace lifetime correction

This checkpoint corrects the HBM workspace peak for staged materialized attention. The compiler previously multiplied the peak's score slots and query rows by `layer_instances × weight_passes`, although those executions are sequential and can reuse the same scratch. The mapping now saves the live counts for one invocation before scaling physical work. The candidate-independent capacity bound uses the same lifetime helper on one invocation before SRAM filtering.

For normalized compact spans, let `q` be query length, `k` KV length, `n` span count, `d` KV stride, `b` batch, `Hq` query heads, and `h` head dimension. The summed KV length is `K = n·k + d·n·(n−1)/2`. Across spans, score slots are `S = Σ(q·K·b·Hq)` and output query rows are `R = Σ(q·n·b·Hq)`. Staged full materialization uses `8·S` bytes for FP32 score plus probability buffers, plus `4·R·h` bytes when output conversion scratch is required (`output_dtype != fp32`). Standalone QK uses `4·S`. These calculations use checked wide intermediates.

`layer_instances` and `weight_passes` still multiply physical operation, byte, and slot counters. They do not multiply the reusable workspace peak. Compact span count and batch remain in `S` and `R`: the task describes one invocation's simultaneous request set. This is why the correction removes only sequential execution factors. The same workspace helper is shared by candidate Plans and the pre-SRAM capacity bound; the latter still does not claim a mapping is executable.

The full-score materialization template and dataflow are `full_score_materialization_v3`. The standalone QK template is `attention_qk_materialized_v3` and uses that dataflow ID. Those Plan summaries carry `workspace_model_revision=workspace_lifetimes_v2`; their Plan identities change. Other attention template/dataflow IDs remain unchanged. The capacity C ABI reports `capacity_model_revision=workspace_lifetimes_v2` while retaining `workspace_capacity_bound_schema_version=1`. LRU certificates use `lru_manifest_workspace_certificate_v2`. Clients reject a v1 capacity response; v1 capacity certificates and affected Plan identities cannot be reused with this model.

## Regression coverage

`coarse_attention_accounting_test` checks a task repeated across two layers and three weight passes: workspace remains 320 bytes, while physical operations, slots, and byte ledgers scale sixfold. It also runs the INT4 odd-tail accounting cases. `coarse_workspace_envelope_test` checks that a repeated child has an 8,704-byte candidate-independent bound equal to its feasible candidate maximum, while the corresponding large parent has a 98,304-byte pre-SRAM bound despite having no materialized candidate after SRAM filtering. Its extreme-repeat case returns a 136-byte one-invocation capacity bound but confirms that physical compile aggregation still fails on overflow. Plan hashes differ across repeat inputs and from the superseded legacy identity.

## Reproducing the public C++ build and tests

From the repository root, activate the configured toolchain and build the standalone public library. CMake finds the JSON header in the checked-in ASTRA-sim tree:

```bash
source /workspace/cloud-setup/activate.sh
cmake -S die -B work/build-attention -DCMAKE_BUILD_TYPE=Release
cmake --build work/build-attention -j2
ctest --test-dir work/build-attention --output-on-failure
```

The captured build and full CTest output are in [public-build-all.log](../../evidence/attention-workspace-20261014/public-build-all.log) and [public-ctest-full.log](../../evidence/attention-workspace-20261014/public-ctest-full.log). The recorded run completed the default build and passed 30/30 CTests in 43.49 seconds, including the attention accounting, compact batch, and workspace envelope tests.

## Reproducing the LRU client and manifest tests

The LRU checkout does not contain generated Chakra Python bindings. [`run_lru_tests.py`](run_lru_tests.py) compiles the LRU checkout's checked-in `et_def.proto` with the real `protoc` into `astra-sim-wscllm-LRU/work/chakra-python-root`. It copies the exact `generate_trace.py` source bytes into that generated work root, so its ordinary root-discovery logic finds the real generated protobuf marker. The runner asserts the imported module path, discovered project root, and copied source equality. It does not patch marker checks or mock generator/router behavior, and it writes no production source files.

Pass the absolute public ABI library path through the environment or `--library`:

```bash
source /workspace/cloud-setup/activate.sh
python3 die/docs/analysis/attention-workspace-20261014/run_lru_tests.py \
  --library "$(realpath work/build-attention/lib/libcoarse_die_cabi.so)"
```

For another checkout location, pass `--repo-root /absolute/path/to/repository`; when omitted, the runner discovers the root relative to its own path. It also accepts `DIE_WORKSPACE_ENVELOPE_TEST_LIBRARY` or `DIE_COARSE_CABI_PATH` as the caller-supplied absolute library path. The captured run executed 19 tests: 18 passed and one archived-library negative test was skipped because its old `.so` was absent. The new strict-v1-rejection test passed against the real v2 C ABI. See [lru-client-manifest-real-protoc.log](../../evidence/attention-workspace-20261014/lru-client-manifest-real-protoc.log).

The real-protoc run made before this portable package runner was added is preserved in [lru-client-manifest-real-protoc-prior-run.log](../../evidence/attention-workspace-20261014/lru-client-manifest-real-protoc-prior-run.log). The first confirmation run of the packaged runner is retained as [lru-client-manifest-real-protoc-package-first-run.log](../../evidence/attention-workspace-20261014/lru-client-manifest-real-protoc-package-first-run.log). Root discovery and working-directory isolation were also verified by running from outside the repository without `--repo-root`; see [lru-client-manifest-external-cwd.log](../../evidence/attention-workspace-20261014/lru-client-manifest-external-cwd.log).

An earlier experimental import harness used a `Path.exists` shim and was discarded as invalid setup; its log was removed. It is not used by this package or counted as test evidence. No claims are made for other production paths, the compiled-GEMM utility, or JOINT native-adapter validation. The five-repository published Coarse Die source digest is recorded in the identity file as source-release context, not as evidence of those integrations.

The exact source, library, toolchain, and log hashes are recorded in [identity.json](../../evidence/attention-workspace-20261014/identity.json).
