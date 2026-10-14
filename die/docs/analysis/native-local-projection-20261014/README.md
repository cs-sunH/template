# New JOINT native local-domain bridge

This additive bridge uses the active `astra-sim-joint`, never the archived
`astra-sim-joint_old`. It connects native projection to the same canonical
`coarse_die::Runtime` used by real local execution. It does not open the sh30
production guard yet.

## Frozen input and ownership

`ProjectionInput.coarse_die_domains` is an optional tuple of
`CoarseDieDomainSpec(resource_key, hardware_config)`. On the wire it is an array
of `{resource_key, hardware_config}`. Each key must name an existing HBM resource
whose raw capacity mirrors the canonical descriptor. Both baseline and candidate
worlds retain these frozen domain definitions when applying a private overlay.
The old input wire omits this field completely.

A `DiePlanDescriptor` with explicit `coarse_die_v2` hardware accepts only typed
`coarse_die_task_v2` specs. Native parsing compiles canonical immutable Plans; it
does not translate them into legacy A3 timers or scalar memory service units.
One leg owns a sequential Plan chain, one fair owner and one native completion.
A successor Plan is submitted at its predecessor's actual terminal tick, even
when an outer advance spans several Plan completions.

An external local access uses `TaskSpec.coarse_endpoint`, the unchanged canonical
endpoint object with transaction/endpoint/owner/direction/path/payload fields.
Its scalar `bytes_amount`, `base_ns`, `duration_ns` and latency aliases stay zero.
The Plan and every explicit endpoint on a physical domain share one Runtime,
including its credit constraints, grouped fairness, resources and served/busy
counters. The old scalar HBM allocator never services the same access.

Native gates and graph dependencies still own admission and external joins.
They release only on the outer coarse terminal. Endpoint and Plan identities
are validated before graph/gate mutation. The wrapper retires terminal Runtime
handles, retains bounded per-leg progress for inspection, emits a completion
once, and refuses to recycle a leg ID before its pending completion is consumed.
Cancellation preserves already served work and stops the remaining chain.

## Verified scope and deliberate boundaries

The native test compares real compiled Plans with direct canonical Runtime runs:
solo, a known delayed restore write, sequential mixed-operation chains,
concurrent candidate/background Plans, baseline/candidate isolation, and the
native slot/dependency completion join. Terminal nanoseconds and all Runtime
counters must agree exactly in these fixtures. Independent wrapper tests cover
cancel/fork/failed-input state preservation and deterministic same-tick chains.
The Python integration test crosses both actual C ABIs for the same Plan and
checks legacy wire omission and rejection of scalar endpoint aliases.

This first profile is `fresh_integer_local_domains_v1`. It starts from empty
local Runtimes and supports exact integer native timestamps through 2^53-1 ns,
the intersection with the existing native world's double clock. It rejects
legacy active/scalar remaining state, prediction-only eta/gamma/service factors,
untyped streams, and legacy block pipelines rather than sharing an access
between two timing models. Existing legacy worlds retain their old behavior.

The wrapper's in-process `fork_snapshot()` copies Runtime, chain bindings and
completion queues. `export_runtime_snapshot()` is the complete canonical
Runtime snapshot **only**; it is not a wrapper/world wire restoration contract.
Results explicitly carry `live_world_import_supported=false`. Network/pool
ownership, full native graph/world import, ledger projection, known cancellation
replay, and the sh30 provider remain the next integration dependencies.

Projection parity is consistency within the selected model. It does not certify
GEMM granularity, target hardware accuracy, TX8 calibration or host overhead.
The original E5 3%/5% failures and target-data absence remain unchanged. No Python
latency formula, fitted architecture constant or tile-expanded production loop
was added.

## Rebuild and checks

Use the repository's aggregate analytical CMake scaffold, with the active JOINT
root, Release, `BUILDTARGET=congestion_aware`, and
`NETWORK_BACKEND_BUILD_AS_LIBRARY=ON`. Build `joint_projection`,
`NativeProjectionCApiTest`, `NativeCoarseProjectionTest`, and
`NativeCoarseDieDomainTest`; then select those three CTests. The aggregate also
builds the actual Online target and both Die C ABIs separately.

From the JOINT llama2 workload directory, use explicit actual library paths:

```sh
NATIVE_COARSE_PROJECTION_LIBRARY=/absolute/build/lib/libjoint_projection.so \
DIE_COARSE_CABI_PATH=/absolute/build/lib/libcoarse_die_cabi.so \
python3 -m unittest joint.test_native_coarse_projection -v
```

The actual source/binary identity, commands, initial failed overlay test and
successful final checks are recorded in the corresponding evidence directory.
This fixture scope is separate from the real production Online experiments.
